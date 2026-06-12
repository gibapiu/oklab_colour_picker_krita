"""Controller state and Krita foreground synchronization."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol, Sequence

import numpy as np

from oklab_colour_picker.domain.colour_presentation import (
    ColourPresenter,
    PresentedColour,
    default_colour_presenter,
)
from oklab_colour_picker.domain.gamut_fallback import FallbackStrategy
from oklab_colour_picker.domain.colour_state import (
    ColourIntent,
    normalize_oklab_for_krita as _normalize_oklab_for_krita,
)


class ChangeKind(Enum):
    """Why the controller's colour state changed.

    ``kind`` is invormational for views that need it, not a source of truth to bypass a view.
    Echo absorption stays local in each view's state machine.
    """

    PREVIEW = "preview"
    COMMIT = "commit"
    ROLLBACK = "rollback"
    EXTERNAL = "external"
    INITIAL = "initial"


@dataclass(frozen=True)
class ColourSnapshot:
    """Published colour read model."""

    colour: PresentedColour
    kind: ChangeKind

    @property
    def intent(self) -> ColourIntent:
        return self.colour.intent


ColourListener = Callable[[ColourSnapshot], None]
FallbackStrategyProvider = Callable[[ColourIntent], FallbackStrategy]
LOGGER = logging.getLogger(__name__)
LOCAL_INTERACTION_SYNC_GRACE_SECONDS = 0.75


class ForegroundAdapter(Protocol):
    def set_foreground(self, oklab: Sequence[float]) -> np.ndarray | None:
        ...

    def get_foreground(self) -> np.ndarray | None:
        ...


class CommitScheduler(Protocol):
    def call_soon(self, callback: Callable[[], None]) -> None:
        ...


class ForegroundTimer(Protocol):
    def start(self, callback: Callable[[], None]) -> None:
        ...

    def stop(self) -> None:
        ...


class ImmediateScheduler:
    """Synchronous fallback for tests or non-Qt hosts."""

    def call_soon(self, callback: Callable[[], None]) -> None:
        callback()


class ColourPickerController:
    """Own colour state and all foreground reads/writes through an adapter."""

    def __init__(
        self,
        adapter: ForegroundAdapter,
        *,
        scheduler: CommitScheduler | None = None,
        foreground_timer: ForegroundTimer | None = None,
        presenter: ColourPresenter | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._adapter = adapter
        self._scheduler = scheduler if scheduler is not None else ImmediateScheduler()
        self._foreground_timer = foreground_timer
        self._presenter = presenter or default_colour_presenter()
        self._fallback_strategy_provider: FallbackStrategyProvider | None = None
        self._colour_listeners: list[ColourListener] = []
        self._selected_intent: ColourIntent | None = None
        self._pending_commit: ColourIntent | None = None
        self._selection_before_pending_commit: ColourIntent | None = None
        self._commit_scheduled = False
        self._commit_token = 0
        self._last_committed_token: int | None = None
        self._last_committed_colour: np.ndarray | None = None
        self._clock = clock
        self._local_interaction_deadline: float | None = None

        if self._foreground_timer is not None:
            self._foreground_timer.start(self.sync_external_foreground)

    @property
    def selected_colour(self) -> np.ndarray | None:
        return None if self._selected_intent is None else self._selected_intent.paint_oklab

    @property
    def selected_intent(self) -> ColourIntent | None:
        return self._selected_intent

    def add_colour_listener(self, listener: ColourListener) -> None:
        self._colour_listeners.append(listener)
        if self._selected_intent is None:
            self._acquire_foreground()
        if self._selected_intent is None:
            return
        try:
            listener(self._snapshot(self._selected_intent, ChangeKind.INITIAL))
        except Exception:
            LOGGER.exception("colour listener failed")

    def remove_colour_listener(self, listener: ColourListener) -> None:
        try:
            self._colour_listeners.remove(listener)
        except ValueError:
            pass

    def set_fallback_strategy_provider(
        self, provider: FallbackStrategyProvider | None
    ) -> None:
        """Set the policy that resolves each colour's out-of-gamut fallback.

        The provider is queried per presented colour.
        """

        self._fallback_strategy_provider = provider

    def reproject(self) -> None:
        """Re-present the current colour unchanged - use when fallback policy changed but the colour did not."""

        if self._selected_intent is not None:
            self._broadcast(self._selected_intent, ChangeKind.PREVIEW)

    def _broadcast(self, intent: ColourIntent, kind: ChangeKind) -> None:
        """Notify every listener uniformly (no skip-the-originator logic).

        Each view's state machine decides whether to honour or absorb the inbound colour.
        """

        snapshot = self._snapshot(intent, kind)
        for listener in list(self._colour_listeners):
            try:
                listener(snapshot)
            except Exception:
                LOGGER.exception("colour listener failed")

    def _snapshot(self, intent: ColourIntent, kind: ChangeKind) -> ColourSnapshot:
        return ColourSnapshot(self._present(intent), kind)

    def _present(self, intent: ColourIntent) -> PresentedColour:
        if self._fallback_strategy_provider is None:
            return self._presenter.present(intent)
        strategy = self._fallback_strategy_provider(intent)
        return self._presenter.with_fallback_strategy(strategy).present(intent)

    def set_preview_colour(self, oklab: ColourIntent | Sequence[float] | None) -> None:
        """Set transient UI preview state without replacing any pending commit.

        Broadcasts ``PREVIEW`` so *other* views can track a mid-drag preview.
        The emitting view self-absorbs the echo via its own state machine.
        """

        self._selected_intent = None if oklab is None else self._intent_from_value(oklab)
        if oklab is None:
            return
        self._extend_local_interaction_guard()
        self._broadcast(self._selected_intent, ChangeKind.PREVIEW)

    def request_foreground_commit(self, oklab: ColourIntent | Sequence[float] | None) -> None:
        if oklab is None:
            return

        intent = self._intent_from_value(oklab)
        if self._pending_commit is None:
            self._selection_before_pending_commit = self._selected_intent
        self._selected_intent = intent
        self._pending_commit = intent
        self._extend_local_interaction_guard()
        if self._commit_scheduled:
            return

        self._commit_scheduled = True
        self._scheduler.call_soon(self._flush_pending_commit)

    def sync_external_foreground(self, *, force: bool = False) -> bool:
        seeding = self._selected_intent is None
        intent = self._acquire_foreground(force=force)
        if intent is None:
            return False
        self._broadcast(intent, ChangeKind.INITIAL if seeding else ChangeKind.EXTERNAL)
        return True

    def _acquire_foreground(self, *, force: bool = False) -> ColourIntent | None:
        """Read the adapter foreground into state without broadcasting.

        Single idempotent acquisition primitive shared by the poll timer,
        canvasChanged, and the synchronous subscribe-time pull.
        """

        if self._local_interaction_blocks_external_sync(force=force):
            return None

        foreground = self._adapter.get_foreground()
        if foreground is None:
            return None

        intent = self._intent_from_value(foreground)
        normalized = normalize_oklab_for_krita(intent.paint_oklab)
        if self._is_self_feedback(normalized):
            return None
        selected_normalized = (
            None if self._selected_intent is None else self._selected_intent.quantized_paint_oklab
        )
        if selected_normalized is not None and _quantized_equal(selected_normalized, normalized):
            return None

        self._selected_intent = intent
        if self._pending_commit is not None:
            self._selection_before_pending_commit = intent
        self._last_committed_token = None
        self._last_committed_colour = None
        return intent

    def _flush_pending_commit(self) -> None:
        self._commit_scheduled = False
        intent = self._pending_commit
        self._pending_commit = None
        selection_before_commit = self._selection_before_pending_commit
        self._selection_before_pending_commit = None
        self._local_interaction_deadline = None
        if intent is None:
            return

        paint_intent = self._resolve_for_paint(intent)
        normalized = paint_intent.quantized_paint_oklab
        if self._last_committed_colour is not None and _quantized_equal(normalized, self._last_committed_colour):
            self._selected_intent = paint_intent
            self._broadcast(
                paint_intent.with_krita_paint_oklab(self._last_committed_colour),
                ChangeKind.COMMIT,
            )
            return

        committed = self._adapter.set_foreground(paint_intent.paint_oklab)
        if committed is None:
            restored = selection_before_commit
            self._selected_intent = restored
            if restored is None:
                return
            self._broadcast(restored, ChangeKind.ROLLBACK)
            return

        self._commit_token += 1
        self._last_committed_token = self._commit_token
        self._last_committed_colour = normalize_oklab_for_krita(committed)
        self._selected_intent = paint_intent
        self._broadcast(
            paint_intent.with_krita_paint_oklab(self._last_committed_colour),
            ChangeKind.COMMIT,
        )

    def _resolve_for_paint(self, intent: ColourIntent) -> ColourIntent:
        """Return the colour to paint for ``intent``: itself when in gamut, else its fallback."""

        presented = self._present(intent)
        if presented.in_gamut:
            return intent
        return presented.fallback.resolved

    def _is_self_feedback(self, normalized_colour: np.ndarray) -> bool:
        return (
            self._last_committed_token == self._commit_token
            and self._last_committed_colour is not None
            and _quantized_equal(normalized_colour, self._last_committed_colour)
        )

    def _extend_local_interaction_guard(self) -> None:
        self._local_interaction_deadline = self._clock() + LOCAL_INTERACTION_SYNC_GRACE_SECONDS

    def _intent_from_value(self, value: ColourIntent | Sequence[float]) -> ColourIntent:
        fallback_hue = 0.0 if self._selected_intent is None else self._selected_intent.hue
        return ColourIntent.from_value(value, achromatic_hue=fallback_hue)

    def _local_interaction_blocks_external_sync(self, *, force: bool = False) -> bool:
        if self._pending_commit is not None or self._commit_scheduled:
            return True
        if force:
            return False
        if self._local_interaction_deadline is None:
            return False
        if self._clock() < self._local_interaction_deadline:
            return True
        self._local_interaction_deadline = None
        return False


def normalize_oklab_for_krita(oklab: Sequence[float]) -> np.ndarray:
    """Normalize OKLab through Krita's 8-bit sRGB foreground precision."""

    return _normalize_oklab_for_krita(oklab)


def _quantized_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare colours already returned by ``normalize_oklab_for_krita``."""

    return bool(np.array_equal(left, right))

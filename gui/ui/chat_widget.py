"""Chat message list: scroll area, message alignment, jump-to-latest pill."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QBoxLayout,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.message_widget import LoadingBubble, MessageBubble

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"


class ChatWidget(QWidget):
    """The scrolling conversation area (messages + welcome overlay)."""

    suggestion_clicked = Signal(str)
    feedback_requested = Signal(str, str)  # (question_id, rating)
    speak_requested = Signal(str)  # assistant answer to read aloud

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("chatContainer")
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(0)
        self._grid.setRowStretch(0, 1)
        self._grid.setColumnStretch(0, 1)

        # Fixed, very light brand watermark behind the transparent scroll
        # surface. Message bubbles remain opaque and readable above it.
        self._watermark_source = QPixmap(str(ASSET_DIR / "nmu_logo_watermark.png"))
        self._watermark = QLabel()
        self._watermark.setObjectName("chatWatermark")
        self._watermark.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._watermark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        watermark_opacity = QGraphicsOpacityEffect(self._watermark)
        watermark_opacity.setOpacity(0.055)
        self._watermark.setGraphicsEffect(watermark_opacity)
        self._grid.addWidget(
            self._watermark, 0, 0, alignment=Qt.AlignmentFlag.AlignCenter
        )

        # --- scroll area ---
        self._scroll = QScrollArea()
        self._scroll.setObjectName("chatScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._scroll.viewport().setAutoFillBackground(False)
        self._scroll.viewport().setAttribute(
            Qt.WidgetAttribute.WA_TranslucentBackground
        )
        self._scroll.verticalScrollBar().setSingleStep(32)
        self._scroll.verticalScrollBar().setPageStep(280)

        container = QWidget()
        container.setObjectName("chatContainer")
        container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        container.setAutoFillBackground(False)
        container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._messages = QVBoxLayout(container)
        self._messages.setContentsMargins(20, 16, 20, 20)
        self._messages.setSpacing(14)
        self._messages.setStretch(self._messages.count(), 1)
        self._messages.addStretch(1)

        self._scroll.setWidget(container)
        self._grid.addWidget(self._scroll, 0, 0)

        # --- jump-to-latest pill (overlay, hidden unless user scrolled up) ---
        self._jump = QPushButton("Jump to latest ↓")
        self._jump.setObjectName("jumpButton")
        self._jump.setCursor(Qt.CursorShape.PointingHandCursor)
        self._jump.setToolTip("Return to the newest message")
        self._jump.clicked.connect(self._jump_to_bottom)
        self._jump.hide()
        self._jump.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed
        )
        self._grid.addWidget(
            self._jump, 0, 0,
            alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
        )

        # --- welcome overlay ---
        from ui.welcome_widget import WelcomeWidget

        self._welcome = WelcomeWidget()
        self._grid.addWidget(self._welcome, 0, 0)
        self._welcome.suggestion_clicked.connect(self.suggestion_clicked)

        self._near_bottom_threshold = 100
        self._near_bottom = True
        bar = self._scroll.verticalScrollBar()
        bar.rangeChanged.connect(self._on_range_changed)
        bar.valueChanged.connect(self._on_scroll)

        self._scroll_pending = False

        self.resizeEvent = self._on_resize
        self._adjust_margins()

    # -- responsive margins ----------------------------------------------
    def _on_resize(self, event) -> None:
        super().resizeEvent(event)
        self._adjust_margins()

    def _adjust_margins(self) -> None:
        w = self.width()
        if w < 900:
            lm, tm, rm, bm = 12, 10, 12, 12
            sp = 12
        elif w < 1400:
            lm, tm, rm, bm = 20, 16, 20, 20
            sp = 14
        else:
            lm, tm, rm, bm = 32, 24, 32, 28
            sp = 18
        self._messages.setContentsMargins(lm, tm, rm, bm)
        self._messages.setSpacing(sp)
        self._adjust_watermark(w)

    def _adjust_watermark(self, width: int) -> None:
        if self._watermark_source.isNull():
            self._watermark.hide()
            return
        if width < 900:
            size = 230
        elif width < 1400:
            size = 320
        else:
            size = 380
        self._watermark.setPixmap(
            self._watermark_source.scaled(
                size,
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._watermark.show()

    # -- message management ----------------------------------------------
    def clear(self) -> None:
        while self._messages.count() > 1:
            item = self._messages.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            layout = item.layout()
            if layout is not None:
                while layout.count():
                    child = layout.takeAt(0)
                    cw = child.widget()
                    if cw is not None:
                        cw.deleteLater()
        self._show_welcome()
        self._schedule_scroll_to_bottom()

    def add_user(self, text: str) -> None:
        self._hide_welcome()
        self._add_bubble(MessageBubble("user", text), align_right=True)

    def add_assistant(self, text: str, sources=None, question_id: str = "") -> None:
        self._hide_welcome()
        bubble = MessageBubble(
            "assistant", text, sources, question_id=question_id
        )
        bubble.feedback_requested.connect(self.feedback_requested.emit)
        bubble.speak_requested.connect(self.speak_requested.emit)
        self._add_bubble(bubble, align_right=False)

    def add_error(self, text: str) -> None:
        self._hide_welcome()
        self._add_bubble(MessageBubble("assistant", text, is_error=True), align_right=False)

    def add_loading(self) -> LoadingBubble:
        self._hide_welcome()
        bubble = LoadingBubble()
        self._add_bubble(bubble, align_right=False)
        return bubble

    def remove_bubble(self, bubble: QWidget) -> None:
        for i in range(self._messages.count()):
            item = self._messages.itemAt(i)
            if item is None:
                continue
            layout = item.layout()
            if layout is not None:
                for j in range(layout.count()):
                    child = layout.itemAt(j)
                    if child is not None and child.widget() is bubble:
                        self._messages.removeItem(item)
                        while layout.count():
                            c = layout.takeAt(0)
                            cw = c.widget()
                            if cw is not None and cw is not bubble:
                                cw.setParent(None)
                        bubble.deleteLater()
                        del layout
                        return
            widget = item.widget()
            if widget is bubble:
                self._messages.removeItem(item)
                bubble.deleteLater()
                return

    def _add_bubble(self, bubble: QWidget, align_right: bool) -> None:
        bubble.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Maximum,
        )
        row = QHBoxLayout()
        # Keep the conversation columns stable regardless of the text's RTL
        # direction: assistant bubbles always share one left edge, while user
        # bubbles always share the right edge. The bubble itself still lays
        # Arabic text out RTL internally.
        row.setDirection(QBoxLayout.Direction.LeftToRight)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        if align_right:
            row.addStretch(1)
            row.addWidget(bubble, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        else:
            row.addWidget(bubble, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            row.addStretch(1)
        self._messages.insertLayout(self._messages.count() - 1, row)
        self._schedule_scroll_to_bottom()

    # -- smooth scrolling -------------------------------------------------
    def _schedule_scroll_to_bottom(self) -> None:
        """Defer scroll so Qt has time to relayout each ReadOnlyText doc."""
        if self._near_bottom:
            if self._scroll_pending:
                return
            self._scroll_pending = True
            QTimer.singleShot(0, self._pulse_scroll)
            QTimer.singleShot(50, self._pulse_scroll)
            QTimer.singleShot(120, self._pulse_scroll)
        else:
            self._jump.show()

    def _pulse_scroll(self) -> None:
        bar = self._scroll.verticalScrollBar()
        bar.setValue(bar.maximum())
        self._scroll_pending = False
        self._jump.hide()

    def _jump_to_bottom(self) -> None:
        self._near_bottom = True
        self._schedule_scroll_to_bottom()

    def _on_range_changed(self, _min: int, _max: int) -> None:
        if self._near_bottom:
            self._schedule_scroll_to_bottom()

    def _on_scroll(self, value: int) -> None:
        bar = self._scroll.verticalScrollBar()
        self._near_bottom = value >= (bar.maximum() - self._near_bottom_threshold)
        if self._near_bottom:
            self._jump.hide()

    # -- welcome overlay --------------------------------------------------
    def _show_welcome(self) -> None:
        self._welcome.show()

    def _hide_welcome(self) -> None:
        self._welcome.hide()

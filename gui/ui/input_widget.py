"""Chat input area: multi-line input, send button, RTL-aware typing."""

from __future__ import annotations

from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from utils.language import is_rtl


class ChatInput(QTextEdit):
    """Enter sends, Shift+Enter inserts a newline, empty input never sends."""

    submitted = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("chatInput")
        self.setPlaceholderText("Ask something about New Mansoura University…")
        self.setAcceptRichText(False)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        self.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.textChanged.connect(self._sync_direction)
        self.textChanged.connect(self._adjust_height)
        self._adjust_height()

    def keyPressEvent(self, event) -> None:  # noqa: D102
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (
            event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self.submitted.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def _sync_direction(self) -> None:
        rtl = is_rtl(self.toPlainText())
        direction = (
            Qt.LayoutDirection.RightToLeft if rtl else Qt.LayoutDirection.LeftToRight
        )
        self.setLayoutDirection(direction)
        target = Qt.AlignmentFlag.AlignRight if rtl else Qt.AlignmentFlag.AlignLeft
        if int(self.alignment()) != int(target):
            self.setAlignment(target)

    def _adjust_height(self) -> None:
        doc = self.document()
        doc_h = doc.size().height()
        min_h = 48
        max_h = 160
        target = max(min_h, min(int(doc_h) + 22, max_h))
        self.setMinimumHeight(target)
        self.setMaximumHeight(target)
        self.updateGeometry()

    def sizeHint(self) -> QSize:  # noqa: D102
        hint = super().sizeHint()
        return QSize(hint.width(), self.minimumHeight() or 48)

    def resizeEvent(self, event) -> None:  # noqa: D102
        super().resizeEvent(event)
        self._adjust_height()


class InputWidget(QWidget):
    """Bottom input bar: text box + Send button + hint line."""

    submitted = Signal(str)  # carries the trimmed message text

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("inputArea")
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 8)
        layout.setSpacing(4)

        row = QHBoxLayout()
        row.setSpacing(8)

        self._input = ChatInput()
        self._input.submitted.connect(self._on_submit)
        self._input.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )

        self._send = QPushButton("Send")
        self._send.setObjectName("sendButton")
        self._send.setCursor(Qt.CursorShape.PointingHandCursor)
        self._send.setToolTip("Send message (Enter)")
        self._send.setSizePolicy(
            QSizePolicy.Policy.Minimum,
            QSizePolicy.Policy.Minimum,
        )
        self._send.clicked.connect(self._on_submit)

        row.addWidget(self._input, stretch=1)
        row.addWidget(self._send, 0, Qt.AlignmentFlag.AlignBottom)
        layout.addLayout(row)

        hint = QLabel("Enter to send  ·  Shift + Enter for a new line")
        hint.setObjectName("inputHint")
        layout.addWidget(hint)

        self._input.textChanged.connect(self._update_send_state)
        self._send.setEnabled(False)

    # -- public API -------------------------------------------------------
    def text(self) -> str:
        return self._input.toPlainText()

    def clear(self) -> None:
        self._input.clear()

    def focus_input(self) -> None:
        self._input.setFocus()

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable sending (disabled while a request is running)."""
        self._enabled = enabled
        self._update_send_state()

    def _update_send_state(self) -> None:
        has_text = bool(self._input.toPlainText().strip())
        self._send.setEnabled(has_text and getattr(self, "_enabled", True))

    def _on_submit(self) -> None:
        text = self._input.toPlainText().strip()
        if not text:
            return
        self.submitted.emit(text)
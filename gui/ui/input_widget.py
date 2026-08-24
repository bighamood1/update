"""Chat input area: multi-line input, send button, RTL-aware typing."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QIcon, QTextCursor
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

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"


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
    """Bottom input bar: text box + microphone + Send + status line."""

    submitted = Signal(str)  # carries the trimmed message text
    voice_requested = Signal()

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

        self._voice = QPushButton()
        self._voice.setObjectName("voiceButton")
        self._voice.setCursor(Qt.CursorShape.PointingHandCursor)
        self._voice.setToolTip("Ask by voice")
        self._voice.setAccessibleName("Ask by voice")
        self._voice.setFixedSize(52, 52)
        self._voice.setIcon(QIcon(str(ASSET_DIR / "microphone.svg")))
        self._voice.setIconSize(QSize(22, 22))
        self._voice.clicked.connect(self.voice_requested.emit)
        self._voice_state = "idle"

        row.addWidget(self._input, stretch=1)
        row.addWidget(self._voice, 0, Qt.AlignmentFlag.AlignBottom)
        row.addWidget(self._send, 0, Qt.AlignmentFlag.AlignBottom)
        layout.addLayout(row)

        self._hint = QLabel("Enter to send  ·  Shift + Enter for a new line  ·  Voice question")
        self._hint.setObjectName("inputHint")
        layout.addWidget(self._hint)

        self._input.textChanged.connect(self._update_send_state)
        self._send.setEnabled(False)

    # -- public API -------------------------------------------------------
    def text(self) -> str:
        return self._input.toPlainText()

    def clear(self) -> None:
        self._input.clear()

    def set_text(self, text: str) -> None:
        self._input.setPlainText(text)
        self._input.moveCursor(QTextCursor.MoveOperation.End)

    def focus_input(self) -> None:
        self._input.setFocus()

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable sending (disabled while a request is running)."""
        self._enabled = enabled
        self._update_send_state()

    def set_voice_enabled(self, enabled: bool) -> None:
        self._voice_available = enabled
        self._update_send_state()

    def set_voice_state(self, state: str, message: str = "") -> None:
        self._voice_state = state
        if state == "listening":
            self._voice.setText("")
            self._voice.setIcon(QIcon(str(ASSET_DIR / "stop_recording.svg")))
            self._voice.setObjectName("voiceButtonRecording")
            self._voice.setToolTip("Stop recording")
            self._voice.setAccessibleName("Stop recording")
            hint = message or "Listening… speak now. Recording stops after a short silence."
        elif state == "transcribing":
            self._voice.setText("…")
            self._voice.setIcon(QIcon())
            self._voice.setObjectName("voiceButton")
            self._voice.setToolTip("Transcribing locally")
            hint = message or "Transcribing your voice locally…"
        elif state == "speaking":
            self._voice.setText("")
            self._voice.setIcon(QIcon(str(ASSET_DIR / "microphone.svg")))
            self._voice.setObjectName("voiceButton")
            self._voice.setToolTip("Ask another question by voice")
            hint = message or "Speaking answer… click its speaker button to stop or replay."
        elif state == "error":
            self._voice.setText("")
            self._voice.setIcon(QIcon(str(ASSET_DIR / "microphone.svg")))
            self._voice.setObjectName("voiceButton")
            self._voice.setToolTip("Ask by voice")
            hint = message or "Voice input failed. Please try again."
        else:
            self._voice.setText("")
            self._voice.setIcon(QIcon(str(ASSET_DIR / "microphone.svg")))
            self._voice.setObjectName("voiceButton")
            self._voice.setToolTip("Ask by voice")
            self._voice.setAccessibleName("Ask by voice")
            hint = message or "Enter to send  ·  Shift + Enter for a new line  ·  Voice question"
        self._voice.style().unpolish(self._voice)
        self._voice.style().polish(self._voice)
        self._hint.setText(hint)
        self._update_send_state()

    def _update_send_state(self) -> None:
        has_text = bool(self._input.toPlainText().strip())
        enabled = getattr(self, "_enabled", True)
        self._send.setEnabled(has_text and enabled)
        available = getattr(self, "_voice_available", True)
        # While recording the button must stay active so it can stop. During
        # transcription or a RAG request, accepting a second utterance would
        # create overlapping turns, so it is disabled.
        self._voice.setEnabled(
            available and (self._voice_state == "listening" or (
                enabled and self._voice_state != "transcribing"
            ))
        )

    def _on_submit(self) -> None:
        text = self._input.toPlainText().strip()
        if not text:
            return
        self.submitted.emit(text)

"""Main application window assembling header, chat, and input."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from api_client import APIClient, ChatResult
from config import API_BASE_URL, API_TIMEOUT_SECONDS
from styles.theme import COLORS, FONT_FAMILY, build_qss
from ui.chat_widget import ChatWidget
from ui.input_widget import InputWidget
from ui.message_widget import LoadingBubble
from worker import ApiWorker, FeedbackWorker

GUIDIR = Path(__file__).resolve().parents[1]


class ChatWindow(QMainWindow):
    """Top-level window for the NMU AI Assistant desktop client."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NMU AI Assistant — New Mansoura University")
        self.resize(1100, 760)
        self.setMinimumSize(820, 580)

        self._client = APIClient(API_BASE_URL, timeout=API_TIMEOUT_SECONDS)
        self._worker: ApiWorker | None = None
        self._loading_bubble: LoadingBubble | None = None
        self._feedback_workers: list[FeedbackWorker] = []
        # Rolling conversation context (last ~10 turns) sent with each message
        # so follow-up questions ("وماذا عن كلية الطب؟") can resolve references.
        self._history: list[dict] = []

        central = QWidget()
        central.setObjectName("root")
        self.setCentralWidget(central)
        central.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.setStretch(0, 0)
        root.setStretch(1, 1)
        root.setStretch(2, 0)

        root.addWidget(self._build_header())
        self._chat = ChatWidget()
        self._chat.suggestion_clicked.connect(self.send_question)
        self._chat.feedback_requested.connect(self._on_feedback)
        self._chat.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        root.addWidget(self._chat, stretch=1)
        self._input_widget = InputWidget()
        self._input_widget.submitted.connect(self.send_question)
        self._input_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        root.addWidget(self._input_widget)

        self._apply_styles()
        self._maybe_load_icon()
        self._input_widget.focus_input()

    # -- header -----------------------------------------------------------
    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("header")

        layout = QHBoxLayout(header)
        layout.setContentsMargins(18, 12, 18, 12)
        layout.setSpacing(8)

        titles = QVBoxLayout()
        titles.setSpacing(0)
        title = QLabel("NMU AI Assistant")
        title.setObjectName("appTitle")
        subtitle = QLabel("New Mansoura University")
        subtitle.setObjectName("appSubtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        layout.addLayout(titles)

        layout.addStretch(1)

        status = QLabel("●  Local Assistant")
        status.setObjectName("statusDot")
        status.setToolTip("Running fully on this computer")
        layout.addWidget(status)

        clear_btn = QPushButton("Clear Chat")
        clear_btn.setObjectName("clearButton")
        clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_btn.setToolTip("Clear this conversation (local only)")
        clear_btn.clicked.connect(self.clear_chat)
        layout.addWidget(clear_btn)

        return header

    # -- actions ----------------------------------------------------------
    def send_question(self, message: str) -> None:
        """Send a question; called by input submit or a welcome suggestion."""
        message = (message or "").strip()
        if self._busy or not message:
            return

        self._chat.add_user(message)
        self._input_widget.clear()
        self._set_busy(True)

        self._loading_bubble = self._chat.add_loading()

        self._worker = ApiWorker(self._client, message, history=list(self._history))
        self._worker.succeeded.connect(self._on_result)
        self._worker.failed.connect(self._on_error)
        self._worker.start()

    def clear_chat(self) -> None:
        if not self._confirm_clear():
            return
        self._chat.clear()
        self._input_widget.clear()
        self._history = []
        self._input_widget.focus_input()

    def _confirm_clear(self) -> bool:
        box = QMessageBox(self)
        box.setWindowTitle("Clear chat")
        box.setText("Clear this conversation?")
        box.setInformativeText(
            "This only clears the current chat view. No data or RAG state is affected."
        )
        box.setStandardButtons(
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
        )
        # Default is the safe action: chat is never cleared by Enter/Escape.
        box.setDefaultButton(QMessageBox.StandardButton.No)
        yes_btn = box.button(QMessageBox.StandardButton.Yes)
        if yes_btn is not None:
            yes_btn.setText("Clear")
        result = box.exec()
        return result == QMessageBox.StandardButton.Yes

    # -- busy / loading ---------------------------------------------------
    @property
    def _busy(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _set_busy(self, busy: bool) -> None:
        self._input_widget.set_enabled(not busy)

    def _remove_loading(self) -> None:
        # Clear the reference first so a removal error can never leave the
        # loading bubble permanently stuck / re-removable.
        bubble = self._loading_bubble
        self._loading_bubble = None
        if bubble is not None:
            self._chat.remove_bubble(bubble)

    def _on_result(self, result: ChatResult) -> None:
        try:
            self._remove_loading()
            self._chat.add_assistant(
                result.answer,
                result.sources,
                question_id=result.question_id,
            )
            self._history.append({"role": "user", "content": getattr(self._worker, "_message", "") or ""})
            self._history.append({"role": "assistant", "content": result.answer})
            self._history = self._history[-20:]
        finally:
            self._set_busy(False)
            self._input_widget.focus_input()

    def _on_error(self, message: str) -> None:
        try:
            self._remove_loading()
            self._chat.add_error(message)
        finally:
            self._set_busy(False)
            self._input_widget.focus_input()

    def _on_feedback(self, question_id: str, rating: str) -> None:
        """Send a user rating off the UI thread (best-effort, non-blocking)."""
        if not question_id:
            return
        worker = FeedbackWorker(self._client, question_id, rating)
        worker.finished.connect(
            lambda w=worker: self._forget_feedback_worker(w)
        )
        self._feedback_workers.append(worker)
        worker.start()

    def _forget_feedback_worker(self, worker: FeedbackWorker) -> None:
        try:
            worker.wait(200)
            if worker in self._feedback_workers:
                self._feedback_workers.remove(worker)
        except Exception:  # pragma: no cover - cosmetic only
            pass

    # -- polish -----------------------------------------------------------
    def _apply_styles(self) -> None:
        font = QFont(FONT_FAMILY)
        font.setPointSize(10)
        QMainWindow.setFont(self, font)
        self.setStyleSheet(build_qss(COLORS))

    def _maybe_load_icon(self) -> None:
        icon_path = GUIDIR / "assets" / "app_icon.svg"
        if icon_path.exists():
            try:
                self.setWindowIcon(QIcon(str(icon_path)))
            except Exception:  # pragma: no cover - cosmetic only
                pass

    def closeEvent(self, event) -> None:  # noqa: D102
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
        for worker in self._feedback_workers:
            if worker.isRunning():
                worker.wait(2000)
        event.accept()
"""Main application window assembling header, chat, and input."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
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
from config import API_BASE_URL, API_TIMEOUT_SECONDS, VOICE_ENABLED
from styles.theme import COLORS, FONT_FAMILY, build_qss
from ui.chat_widget import ChatWidget
from ui.input_widget import InputWidget
from ui.message_widget import LoadingBubble
from voice import AudioRecorder, SpeechWorker, TranscriptionWorker, VoiceError
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
        self._recorder = AudioRecorder()
        self._transcriber: TranscriptionWorker | None = None
        self._speaker: SpeechWorker | None = None
        self._speech_error: str | None = None
        self._speaking_text = ""
        self._queued_speech = ""
        self._voice_timer = QTimer(self)
        self._voice_timer.setInterval(100)
        self._voice_timer.timeout.connect(self._check_voice_recording)
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
        self._chat.speak_requested.connect(self._speak_answer)
        self._chat.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        root.addWidget(self._chat, stretch=1)
        self._input_widget = InputWidget()
        self._input_widget.submitted.connect(self.send_question)
        self._input_widget.voice_requested.connect(self._toggle_voice_recording)
        self._input_widget.set_voice_enabled(VOICE_ENABLED)
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

        # A new question takes priority over any answer currently being read.
        self._stop_speaking()

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
        self._cancel_voice_activity()
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

    # -- local voice -----------------------------------------------------
    def _toggle_voice_recording(self) -> None:
        """Start/stop one local recording; its transcript uses send_question."""
        if not VOICE_ENABLED or self._busy:
            return
        if self._recorder.recording:
            self._finish_voice_recording()
            return
        self._stop_speaking()
        try:
            self._recorder.start()
        except VoiceError as exc:
            self._voice_error(str(exc))
            return
        self._input_widget.set_voice_state("listening")
        self._voice_timer.start()

    def _check_voice_recording(self) -> None:
        if self._recorder.should_auto_stop():
            self._finish_voice_recording()

    def _finish_voice_recording(self) -> None:
        self._voice_timer.stop()
        try:
            audio_path = self._recorder.stop()
        except VoiceError as exc:
            self._voice_error(str(exc))
            return
        self._input_widget.set_enabled(False)
        self._input_widget.set_voice_state(
            "transcribing", "Transcribing locally… the first use may load the speech model."
        )
        worker = TranscriptionWorker(audio_path)
        self._transcriber = worker
        worker.succeeded.connect(self._on_transcribed)
        worker.failed.connect(self._voice_error)
        worker.finished.connect(self._forget_transcriber)
        worker.start()

    def _on_transcribed(self, text: str) -> None:
        """Display the transcript then send it through the normal RAG path."""
        self._input_widget.set_enabled(True)
        self._input_widget.set_voice_state("idle", f"Heard: {text}")
        self._input_widget.set_text(text)
        # This is intentionally the same method used by the Send button.
        self.send_question(text)

    def _forget_transcriber(self) -> None:
        self._transcriber = None

    def _voice_error(self, message: str) -> None:
        self._voice_timer.stop()
        self._recorder.cancel()
        self._input_widget.set_enabled(True)
        self._input_widget.set_voice_state("error", message)
        self._chat.add_error(message)

    def _speak_answer(self, text: str) -> None:
        """Toggle local speech for one displayed answer (never calls RAG)."""
        text = (text or "").strip()
        if not text:
            return
        if self._speaker is not None and self._speaker.isRunning():
            if self._speaking_text == text:
                self._queued_speech = ""
            else:
                self._queued_speech = text
            self._speaker.cancel()
            return
        self._start_speaking(text)

    def _start_speaking(self, text: str) -> None:
        self._speaking_text = text
        self._input_widget.set_voice_state("speaking")
        worker = SpeechWorker(text)
        self._speaker = worker
        self._speech_error = None
        worker.finished_ok.connect(self._on_speech_finished)
        worker.failed.connect(self._on_speech_failed)
        # Keep the QThread referenced until Qt confirms that run() has fully
        # returned. Dropping the last Python reference from a custom signal
        # can abort the whole process with "QThread: Destroyed while thread is
        # still running".
        worker.finished.connect(
            lambda w=worker: self._on_speech_thread_finished(w)
        )
        worker.start()

    def _on_speech_finished(self) -> None:
        """Record a successful result; cleanup waits for QThread.finished."""
        self._speech_error = None

    def _on_speech_failed(self, message: str) -> None:
        """Record the error; cleanup waits for QThread.finished."""
        self._speech_error = message

    def _on_speech_thread_finished(self, worker: SpeechWorker) -> None:
        """Release a speaker only after Qt confirms its thread has stopped."""
        if self._speaker is not worker:
            worker.deleteLater()
            return
        error = self._speech_error
        self._speaker = None
        self._speech_error = None
        self._speaking_text = ""
        queued, self._queued_speech = self._queued_speech, ""
        worker.deleteLater()
        if error:
            self._voice_error(error)
        elif queued:
            self._start_speaking(queued)
        else:
            self._input_widget.set_voice_state("idle")

    def _stop_speaking(self) -> None:
        self._queued_speech = ""
        if self._speaker is not None and self._speaker.isRunning():
            self._speaker.cancel()
        else:
            self._speaker = None
            self._speaking_text = ""
            self._input_widget.set_voice_state("idle")

    def _cancel_voice_activity(self) -> None:
        self._voice_timer.stop()
        self._recorder.cancel()
        self._stop_speaking()
        self._input_widget.set_voice_state("idle")

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
        self._voice_timer.stop()
        self._recorder.cancel()
        self._stop_speaking()
        active_threads = [
            worker
            for worker in (
                self._transcriber,
                self._worker,
                self._speaker,
                *self._feedback_workers,
            )
            if worker is not None and worker.isRunning()
        ]
        if active_threads:
            # Never destroy the window while a QThread is active. Polling keeps
            # the UI responsive and closes automatically as soon as every
            # background operation has finished.
            self._input_widget.set_voice_state(
                "transcribing", "Finishing the current local operation before closing…"
            )
            event.ignore()
            QTimer.singleShot(150, self.close)
            return
        event.accept()

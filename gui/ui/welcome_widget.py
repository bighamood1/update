"""Empty-state welcome screen with clickable suggested questions."""

from __future__ import annotations

from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

SUGGESTIONS = [
    "What faculties does NMU have?",
    "Where is New Mansoura University located?",
    "ما هي كليات جامعة المنصورة الجديدة؟",
]


class WelcomeWidget(QWidget):
    """Centered welcome panel shown until the first message is sent."""

    suggestion_clicked = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("welcome")
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(10)
        layout.addStretch(1)

        title = QLabel("NMU AI Assistant")
        title.setObjectName("welcomeTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setWordWrap(True)
        layout.addWidget(title)

        subtitle = QLabel("Ask anything about New Mansoura University.")
        subtitle.setObjectName("welcomeSubtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addSpacing(22)

        for question in SUGGESTIONS:
            button = QPushButton(question)
            button.setObjectName("suggestionButton")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setSizePolicy(
                QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Fixed,
            )
            button.setMinimumWidth(260)
            button.clicked.connect(
                lambda _=False, q=question: self.suggestion_clicked.emit(q)
            )
            layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignHCenter)

        layout.addStretch(2)
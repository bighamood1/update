"""Auto-sizing read-only rich-text view for assistant answers.

QTextBrowser gives proper Markdown rendering (headings, bullets, numbered
lists, links) and selectable text without the layout/contrast quirks of a
plain QLabel. Internal scrollbars are disabled and the height tracks the
document so messages grow naturally inside the chat scroll area.
"""

from __future__ import annotations

import re
from typing import Final

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QFont, QTextOption
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from styles.theme import ANSWER_POINT_SIZE, ANSWER_STYLESHEET, FONT_FAMILY, COLORS
from utils.language import direction_for, is_rtl

# ---------------------------------------------------------------------------
# Answer pre-processing: strip inline source markers the LLM might produce
# since the dedicated Sources panel already handles citation display.
# ---------------------------------------------------------------------------

_SOURCE_MARKER_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"\[\s*Source\s*\d+\s*\]", re.IGNORECASE),
    re.compile(r"\[\s*Source\s*\d+\s*(?:,\s*Source\s*\d+\s*)*\]", re.IGNORECASE),
    re.compile(r"\[\s*\d+\s*\]"),
    re.compile(r"\(\s*Source\s*\d+\s*\)", re.IGNORECASE),
    re.compile(r"^\s*Source\s*\d+\s*[:\-]", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*\d+\s*\.\s*\[[^\]]*\]\s*[:\-]", re.MULTILINE),
    re.compile(
        r"^\s*(?:Sources?|References?|Citations?)\s*[:\-]"
        r"(?:\s*\n(?:.+\n)*.*$)?",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(r"\[\s*Source\s*\d+(?:\s*[:\-]\s*[^\]]*)?\s*\]", re.IGNORECASE),
    re.compile(r"\bSource\s+\d+\b\s*[:\-]\s*(?:https?://\S+|www\.\S+)", re.IGNORECASE),
    re.compile(r"\b(?:https?://\S+)\s*\[\s*Source\s*\d+\s*\]", re.IGNORECASE),
]

_URL_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<!\])\b(https?://[^\s<>\"')\]]+|www\.[^\s<>\"')\]]+)",
    re.IGNORECASE,
)

_MD_LINK_RE: Final[re.Pattern[str]] = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

_INLINE_CODE_RE: Final[re.Pattern[str]] = re.compile(r"`([^`]+)`")


def _strip_inline_source_labels(text: str) -> str:
    """Remove inline "[Source 1]", "Source 2:", trailing Sources: blocks, etc."""
    if not text:
        return text
    cleaned = text
    for pat in _SOURCE_MARKER_PATTERNS:
        cleaned = pat.sub("", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def _linkify_plain_urls(text: str) -> str:
    """Convert bare URLs into Markdown links while leaving existing MD links alone."""
    if not text:
        return text

    protected: list[str] = []

    def _stash_md(m):
        idx = len(protected)
        protected.append(m.group(0))
        return f"\x00MD{idx}\x00"

    def _stash_code(m):
        idx = len(protected)
        protected.append(m.group(0))
        return f"\x00CD{idx}\x00"

    temp = _MD_LINK_RE.sub(_stash_md, text)
    temp = _INLINE_CODE_RE.sub(_stash_code, temp)

    def _wrap_url(m):
        url = m.group(1)
        href = url if url.lower().startswith("http") else f"https://{url}"
        label = url
        if len(label) > 80:
            label = label[:77] + "…"
        return f"[{label}]({href})"

    temp = _URL_RE.sub(_wrap_url, temp)

    def _restore(m):
        kind = m.group(1)
        num = int(m.group(2))
        return protected[num] if 0 <= num < len(protected) else m.group(0)

    temp = re.sub(r"\x00(MD|CD)(\d+)\x00", _restore, temp)
    return temp


def _prepare_assistant_markdown(text: str) -> str:
    """Full preprocessing pipeline: strip sources, linkify URLs, clean up."""
    if not text:
        return ""
    cleaned = _strip_inline_source_labels(text)
    cleaned = _linkify_plain_urls(cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


class ReadOnlyText(QTextBrowser):
    def __init__(self, text: str = "", *, markdown: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("answerText")
        self.setReadOnly(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setOpenExternalLinks(True)
        self.setLineWrapMode(QTextBrowser.LineWrapMode.WidgetWidth)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.MinimumExpanding,
        )

        doc = self.document()
        doc.setDocumentMargin(0)
        doc.setDefaultStyleSheet(ANSWER_STYLESHEET)

        font = QFont(FONT_FAMILY)
        font.setPointSize(ANSWER_POINT_SIZE)
        self.setFont(font)

        self._markdown = markdown
        self._raw_text = text
        self.set_content(text)
        doc.contentsChanged.connect(self._sync_height)

    # -- content ----------------------------------------------------------
    def set_content(self, text: str) -> None:
        self._raw_text = text
        if self._markdown:
            prepared = _prepare_assistant_markdown(text)
            try:
                self.document().setMarkdown(prepared)
            except Exception:
                self.setPlainText(prepared)
        else:
            self.setPlainText(text or "")
        self._sync_height()

    # -- direction --------------------------------------------------------
    def set_direction(self, text: str) -> None:
        rtl = is_rtl(text)
        option = QTextOption()
        option.setTextDirection(
            Qt.LayoutDirection.RightToLeft if rtl else Qt.LayoutDirection.LeftToRight
        )
        option.setWrapMode(QTextOption.WrapMode.WordWrap)
        option.setAlignment(Qt.AlignmentFlag.AlignTop)
        # More accurate glyph metrics for connected Arabic script -> correct
        # auto-height so long RTL answers never clip or over-expand the bubble.
        option.setUseDesignMetrics(True)
        self.document().setDefaultTextOption(option)
        self.setLayoutDirection(
            Qt.LayoutDirection.RightToLeft if rtl else Qt.LayoutDirection.LeftToRight
        )
        self._sync_height()

    # -- auto height ------------------------------------------------------
    def _sync_height(self) -> None:
        doc = self.document()
        vp = self.viewport()
        vp_width = max(vp.width() if vp else 0, 200)
        doc.setTextWidth(vp_width)
        ideal = doc.size().height()
        h = max(int(ideal) + 4, 20)
        # Clamp so auto-sizing can never feed Qt an absurd maximum height
        # (the "largest allowed size (16777215,16777215)" warning family).
        h = min(h, 4096)
        self.setMinimumHeight(h)
        self.setMaximumHeight(h + 4)
        self.updateGeometry()

    def sizeHint(self) -> QSize:  # noqa: D102
        hint = super().sizeHint()
        h = self.maximumHeight() or hint.height()
        return QSize(hint.width(), max(h, 24))

    def minimumSizeHint(self) -> QSize:  # noqa: D102
        hint = super().minimumSizeHint()
        return QSize(0, self.minimumHeight() or hint.height())

    def resizeEvent(self, event) -> None:  # noqa: D102
        super().resizeEvent(event)
        self._sync_height()


class Spinner(QFrame):
    """Small rotating-arc loading spinner (no assets needed)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(18, 18)
        self._angle = 0
        self._timer: QTimer | None = None

    def start(self) -> None:
        from PySide6.QtCore import QTimer

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(32)
        self.show()

    def stop(self) -> None:
        if self._timer is not None:
            self._timer.stop()
        self.hide()

    def _tick(self) -> None:
        self._angle = (self._angle + 24) % 360
        self.update()

    def paintEvent(self, event) -> None:  # noqa: D102
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QColor, QPainter, QPen

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(COLORS["accent"]), 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        rect = QRectF(2, 2, self.width() - 4, self.height() - 4)
        painter.drawArc(rect, -self._angle * 16, 270 * 16)
        painter.end()


class FeedbackRow(QWidget):
    """Small "Was this answer useful?" row with one-click ratings.

    Emits ``rating_chosen(question_id, rating)`` once per question. After a
    choice is made the row is locked so a question is rated at most once.
    """

    rating_chosen = Signal(str, str)  # (question_id, rating)

    _LABELS: Final[tuple[tuple[str, str], ...]] = (
        ("useful", "Useful"),
        ("somewhat", "Somewhat"),
        ("not_useful", "Not useful"),
    )

    def __init__(self, question_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._question_id = question_id
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(8)
        label = QLabel("Was this answer helpful?")
        label.setObjectName("feedbackPrompt")
        label.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        layout.addWidget(label)
        for rating, text in self._LABELS:
            btn = QPushButton(text)
            btn.setObjectName("feedbackButton")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
            btn.clicked.connect(lambda _=False, r=rating: self._choose(r))
            layout.addWidget(btn)
        self._buttons: list[QPushButton] = [
            b for b in self.findChildren(QPushButton) if b.objectName() == "feedbackButton"
        ]
        layout.addStretch(1)

    def _choose(self, rating: str) -> None:
        self.rating_chosen.emit(self._question_id, rating)
        for btn in self._buttons:
            btn.setEnabled(False)
        self._show_thanks()

    def _show_thanks(self) -> None:
        thanks = QLabel("Thanks for your feedback.")
        thanks.setObjectName("feedbackThanks")
        thanks.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        parent_layout = self.layout()
        if parent_layout is not None:
            parent_layout.addWidget(thanks)


class MessageBubble(QFrame):
    """A single chat message card (user, assistant, error, or welcome)."""

    feedback_requested = Signal(str, str)  # (question_id, rating)

    MAX_WIDTH_USER = 720
    MAX_WIDTH_ASSISTANT = 860
    MAX_WIDTH_ERROR = 860
    MAX_WIDTH_WELCOME = 640

    def __init__(
        self,
        role: str,
        text: str,
        sources=None,
        *,
        is_error: bool = False,
        is_welcome: bool = False,
        question_id: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        object_name = "bubble_error" if is_error else (
            "bubble_welcome" if is_welcome else f"bubble_{role}"
        )
        self.setObjectName(object_name)
        self.setProperty("role", role)

        if is_error:
            self.setMaximumWidth(self.MAX_WIDTH_ERROR)
        elif is_welcome:
            self.setMaximumWidth(self.MAX_WIDTH_WELCOME)
        elif role == "user":
            self.setMaximumWidth(self.MAX_WIDTH_USER)
        else:
            self.setMaximumWidth(self.MAX_WIDTH_ASSISTANT)

        self.setMinimumWidth(200)

        self.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Maximum,
        )

        direction_text = text if not is_error else ""
        self.setLayoutDirection(direction_for(direction_text))

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 14)
        outer.setSpacing(6)

        if role == "assistant" and not is_error and not is_welcome:
            self._answer = ReadOnlyText(text, markdown=True)
            self._answer.set_direction(text)
            outer.addWidget(self._answer)

            self._sources_panel = None
            self._toggle = None
            self._sources_visible = False
            if sources:
                from ui.source_widget import SourcesPanel
                from PySide6.QtWidgets import QPushButton

                self._toggle = QPushButton("View Sources")
                self._toggle.setObjectName("sourcesToggle")
                self._toggle.setCursor(Qt.CursorShape.PointingHandCursor)
                self._toggle.setToolTip("Show the pages used for this answer")
                self._toggle.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
                self._toggle.clicked.connect(self._on_toggle)
                outer.addWidget(
                    self._toggle, alignment=Qt.AlignmentFlag.AlignLeft
                )

                self._sources_panel = SourcesPanel(sources)
                self._sources_panel.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
                self._sources_panel.setVisible(False)
                outer.addWidget(self._sources_panel)

            if question_id:
                self._feedback_row = FeedbackRow(question_id)
                self._feedback_row.rating_chosen.connect(self.feedback_requested.emit)
                outer.addWidget(
                    self._feedback_row, alignment=Qt.AlignmentFlag.AlignLeft
                )
        else:
            use_md = not is_welcome
            label = ReadOnlyText(text, markdown=use_md)
            label.set_direction(direction_text)
            outer.addWidget(label)

    def _on_toggle(self) -> None:
        assert self._sources_panel is not None and self._toggle is not None
        self._sources_visible = not self._sources_visible
        self._sources_panel.setVisible(self._sources_visible)
        self._toggle.setText("Hide Sources" if self._sources_visible else "View Sources")


class LoadingBubble(QFrame):
    """Animated "Generating answer..." card shown while the backend answers."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("bubble_loading")
        self.setMaximumWidth(MessageBubble.MAX_WIDTH_ASSISTANT)
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Maximum,
        )
        self.setLayoutDirection(Qt.LayoutDirection.LeftToRight)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)

        row = self._build_row()
        layout.addLayout(row)

    def _build_row(self):
        from PySide6.QtWidgets import QHBoxLayout, QLabel

        row = QHBoxLayout()
        row.setSpacing(10)
        self._spinner = Spinner()
        self._spinner.start()
        row.addWidget(self._spinner, 0, Qt.AlignmentFlag.AlignVCenter)
        self._label = QLabel("Generating answer…")
        self._label.setObjectName("loadingLabel")
        self._label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self._label, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addStretch(1)
        return row

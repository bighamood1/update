"""Collapsible source list shown behind the "View Sources" button."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget

from api_client import Source
from styles.theme import COLORS


def _esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _short_label(url: str, limit: int = 80) -> str:
    if len(url) <= limit:
        return url
    head = url[: limit - 40]
    tail = url[-37:]
    return f"{head}…{tail}"


class SourcesPanel(QFrame):
    """Hidden-by-default, expandable list of clickable source links."""

    def __init__(self, sources: list[Source], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sourcesPanel")
        self.setVisible(False)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Maximum,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        heading = QLabel("Sources")
        heading.setObjectName("sourcesHeading")
        layout.addWidget(heading)

        for i, src in enumerate(sources, start=1):
            row = QLabel()
            row.setObjectName("sourceRow")
            row.setWordWrap(True)
            row.setOpenExternalLinks(True)
            row.setTextInteractionFlags(
                Qt.TextInteractionFlag.LinksAccessibleByMouse
                | Qt.TextInteractionFlag.TextSelectableByMouse
            )
            row.setTextFormat(Qt.TextFormat.RichText)
            row.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Preferred,
            )
            row.setStyleSheet(
                f"QLabel {{ color: {COLORS['text_primary']}; line-height: 1.55; }}"
            )
            title = _esc(src.title)
            url = _esc(src.url)
            display_url = _esc(_short_label(src.url)) if src.url else ""
            if src.url:
                row.setText(
                    f'{i}. <b>{title}</b><br/>'
                    f'<span style="font-size:12px;">'
                    f'<a href="{url}" style="color:{COLORS["link"]}; text-decoration:none; word-break:break-all;">'
                    f'{display_url}</a></span>'
                )
            else:
                row.setText(f"{i}. <b>{title}</b>")
            layout.addWidget(row)
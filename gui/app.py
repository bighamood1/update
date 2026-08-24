"""NMU AI Assistant — desktop chat GUI (client only).

Entry point. This GUI is a pure HTTP client: it sends questions to the local
RAG API and shows the final answer plus an optional, user-expandable source
list. It never touches ChromaDB, embeddings, reranking, or Ollama.

Run::

    ..\\.venv\\Scripts\\python.exe app.py
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from ui.main_window import ChatWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("NMU AI Assistant")
    app.setOrganizationName("NMU")

    window = ChatWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
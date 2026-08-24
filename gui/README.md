# NMU AI Assistant — GUI (client)

A clean desktop chat interface for the existing **NMU AI Robot Assistant —
PHASE 2 (Local RAG)** project.

The GUI is **only a client**. It sends questions over HTTP to the local RAG
API and shows the final answer (plus an optional, user-expandable source
list). It does **not** touch ChromaDB, embeddings, reranking, or Ollama, and
it does **not** modify any existing RAG code.

The same interface now accepts both text and voice. Press the microphone,
speak in Arabic or English, and pause (or press stop): local Whisper turns the
recording into text, then that transcript is sent through the exact same
`/chat` endpoint and conversation history as typed input. Press the speaker
button beside any assistant message to read that displayed answer aloud;
playback never generates a second answer or bypasses the RAG data.

```
GUI  ──HTTP──▶  RAG API  ──▶  Existing RAG Pipeline  ──▶  ChromaDB / Ollama / Qwen3-VL:8b
```

## Requirements

- The RAG project must already be set up (`.venv` with `src/rag`, the vector
  index built, and Ollama running with `qwen3-vl:8b`).
- Install the API dependencies (once):

  ```powershell
  .\.venv\Scripts\python.exe -m pip install -r api\requirements.txt
  ```
- Install the GUI dependencies (once):

  ```powershell
  .\.venv\Scripts\python.exe -m pip install -r gui\requirements.txt
  ```

- Windows microphone permission must be enabled for desktop apps. The first
  voice question loads/downloads the configured `faster-whisper` model. The
  first playback of an answer uses the configured neural TTS voice and caches
  the resulting audio locally for instant replay.

## 1. Start the RAG API

From the project root:

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe api\server.py
```

or:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.server:app --host 127.0.0.1 --port 8000
```

The API binds to `127.0.0.1:8000` only (local development). The first request
is slower while the pipeline loads embeddings + reranker; individual answers
then take as long as `qwen3-vl:8b` needs on this CPU (typically 1–8 minutes).

## 2. Start the GUI

```powershell
cd gui
$env:PYTHONIOENCODING='utf-8'
..\.venv\Scripts\python.exe app.py
```

Type a question, press **Enter** to send (`Shift+Enter` = new line). Click
**View Sources** under an answer to expand the sources used (URLs open in
your browser). **Clear chat** resets the session view.

## 3. API specification

Base URL: `http://127.0.0.1:8000`

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /chat`

Request:

```json
{ "message": "What faculties does New Mansoura University have?" }
```

Response:

```json
{
  "answer": "New Mansoura University has the following faculties: ...",
  "sources": [
    { "title": "All Faculties & Programs | NMU",
      "url": "https://nmu.edu.eg/en/all-faculties-programs" }
  ]
}
```

The response contains only the final answer and its sources — no chunk IDs,
scores, embeddings, timings, or retrieval logs.

## 4. Configuration

`gui/config.py` reads settings from the environment (or a `gui/.env` file):

| Variable                | Default                   | Purpose                           |
| ----------------------- | ------------------------- | --------------------------------- |
| `API_BASE_URL`        | `http://127.0.0.1:8000` | Local RAG API endpoint            |
| `API_TIMEOUT_SECONDS` | `1200`                  | Max wait for a long Ollama answer |
| `MAX_ANSWER_CHARS`    | `20000`                 | Defensive answer length cap       |

Example `gui/.env`:

```
API_BASE_URL=http://127.0.0.1:8000
```

## 5. Project layout

```
RAG (PHASE 2)/
├── api/            <- thin local HTTP facade (imports existing RAGPipeline)
│   └── server.py
└── gui/            <- isolated desktop client (HTTP only)
    ├── app.py                  # entry point
    ├── api_client.py           # HTTP client (only backend contact point)
    ├── config.py               # API_BASE_URL + timeouts
    ├── worker.py               # background thread for API calls
    ├── ui/
    │   ├── main_window.py      # header + chat + input assembly
    │   ├── chat_widget.py      # scroll area, alignment, jump-to-latest
    │   ├── message_widget.py   # message cards, answer text, spinner
    │   ├── source_widget.py    # collapsible clickable source list
    │   ├── welcome_widget.py   # empty-state suggestions
    │   └── input_widget.py     # text box + Send (Enter / Shift+Enter)
    ├── utils/
    │   └── language.py         # Arabic/English direction detection
    ├── styles/
    │   └── theme.py            # color palette + stylesheet
    ├── requirements.txt
    ├── README.md
    └── assets/
```

## 6. Design & accessibility

- Light theme with a consistent palette (surface `#FFFFFF`, chat bg
  `#F5F7FA`, primary text `#111827`, accent amber, user bubble `#FFF7ED`).
- Every text/background pair meets WCAG AA contrast (≥ 4.5:1 for body text;
  the Send button uses a darkened amber for a 5.0:1 white-on-orange ratio).
- Arabic detection sets per-message RTL rendering; English/mixed use the
  dominant direction. Arabic input also switches the text box direction.
- Enter sends, Shift+Enter inserts a newline, empty input never sends, the
  Send button is disabled while a request runs, and long answers wrap with
  no horizontal overflow.

## 7. Notes

- The GUI is stateless on the backend side: each `/chat` call is a single
  RAG turn. Chat history is kept only in the current GUI session.
- Error cases (server down, timeout, empty answer, backend failure) are shown
  as friendly messages in the chat — never stack traces or raw JSON.
- No streaming yet; `gui/api_client.py` is structured so a streaming variant
  can be added later without changing the GUI.
- All traffic stays on `127.0.0.1`; nothing is sent to any cloud service.

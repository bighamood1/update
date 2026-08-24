"""Background workers that call the API off the UI thread."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from api_client import (
    APIClient,
    APIConnectionError,
    APIError,
    APIResponseError,
    APIServerError,
    APITimeoutError,
)


class ApiWorker(QThread):
    """Runs one API call and reports the clean result."""

    succeeded = Signal(object)  # ChatResult
    failed = Signal(str)        # user-friendly message

    def __init__(
        self, client: APIClient, message: str, history: list[dict] | None = None
    ) -> None:
        super().__init__()
        self._client = client
        self._message = message
        self._history = history

    def run(self) -> None:  # noqa: D102
        try:
            result = self._client.send_message(self._message, history=self._history)
        except APITimeoutError as exc:
            self.failed.emit(str(exc))
            return
        except APIConnectionError as exc:
            self.failed.emit(str(exc))
            return
        except APIServerError as exc:
            self.failed.emit(str(exc))
            return
        except APIResponseError as exc:
            self.failed.emit(str(exc))
            return
        except APIError as exc:
            self.failed.emit(str(exc))
            return
        except Exception:  # pragma: no cover - absolute safety net
            self.failed.emit(
                "The NMU assistant encountered an unexpected error. Please try again."
            )
            return
        self.succeeded.emit(result)


class FeedbackWorker(QThread):
    """Sends a user rating (useful / medium / not_useful) off the UI thread."""

    finished_ok = Signal(str)      # question_id that was rated
    failed = Signal(str, str)      # question_id, user-friendly message

    def __init__(
        self, client: APIClient, question_id: str, rating: str, reason: str | None = None
    ) -> None:
        super().__init__()
        self._client = client
        self._question_id = question_id
        self._rating = rating
        self._reason = reason

    def run(self) -> None:  # noqa: D102
        try:
            ok = self._client.send_feedback(
                self._question_id, self._rating, self._reason
            )
        except Exception:  # pragma: no cover - absolute safety net
            ok = False
        if ok:
            self.finished_ok.emit(self._question_id)
        else:
            self.failed.emit(
                self._question_id,
                "Your feedback could not be sent. Please check the server.",
            )
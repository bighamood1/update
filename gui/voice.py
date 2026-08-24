"""Local microphone transcription and answer speech for the desktop GUI.

Voice input is deliberately only another input adapter: the transcript is
sent through the same HTTP ``/chat`` endpoint as typed text.  This module does
not search the knowledge base or generate answers itself.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import QThread, Signal

from config import (
    VOICE_MAX_RECORD_SECONDS,
    VOICE_MIN_RECORD_SECONDS,
    VOICE_SAMPLE_RATE,
    VOICE_SILENCE_SECONDS,
    VOICE_SILENCE_THRESHOLD,
    VOICE_STT_COMPUTE_TYPE,
    VOICE_STT_DEVICE,
    VOICE_STT_MODEL,
    VOICE_TTS_ARABIC_VOICE,
    VOICE_TTS_ENGLISH_VOICE,
)


class VoiceError(RuntimeError):
    """A user-facing microphone, transcription, or speech error."""


def contains_arabic(text: str) -> bool:
    return any("\u0600" <= char <= "\u06ff" for char in text or "")


def clean_text_for_speech(text: str) -> str:
    """Turn displayed Markdown into natural plain text for TTS."""
    cleaned = text or ""
    cleaned = re.sub(r"```.*?```", " ", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    cleaned = re.sub(r"!\[([^]]*)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)
    cleaned = re.sub(r"^\s{0,3}#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"[*_~>]", "", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    return cleaned.strip()


class AudioRecorder:
    """Small sounddevice recorder with silence and maximum-duration guards."""

    def __init__(
        self,
        *,
        sample_rate: int = VOICE_SAMPLE_RATE,
        silence_seconds: float = VOICE_SILENCE_SECONDS,
        max_seconds: float = VOICE_MAX_RECORD_SECONDS,
        min_seconds: float = VOICE_MIN_RECORD_SECONDS,
        silence_threshold: float = VOICE_SILENCE_THRESHOLD,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.sample_rate = sample_rate
        self.silence_seconds = silence_seconds
        self.max_seconds = max_seconds
        self.min_seconds = min_seconds
        self.silence_threshold = silence_threshold
        self._clock = clock
        self._stream = None
        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._started_at = 0.0
        self._last_voice_at = 0.0
        self._heard_voice = False

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        if self.recording:
            return
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise VoiceError(
                "Voice input is not installed. Install the GUI voice requirements first."
            ) from exc

        self._frames = []
        self._started_at = self._clock()
        self._last_voice_at = self._started_at
        self._heard_voice = False
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                callback=self._on_audio,
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            raise VoiceError(
                "Could not open the microphone. Check Windows microphone permission "
                "and the selected input device."
            ) from exc

    def _on_audio(self, indata, _frames, _time_info, status) -> None:
        if status:
            # PortAudio status flags are transient; keep the usable frames.
            pass
        frame = np.asarray(indata, dtype=np.int16).copy()
        if frame.size:
            samples = frame.astype(np.float32)
            rms = math.sqrt(float(np.mean(samples * samples)))
            now = self._clock()
            if rms >= self.silence_threshold:
                self._heard_voice = True
                self._last_voice_at = now
            with self._lock:
                self._frames.append(frame)

    def should_auto_stop(self) -> bool:
        if not self.recording:
            return False
        now = self._clock()
        elapsed = now - self._started_at
        if elapsed >= self.max_seconds:
            return True
        return (
            self._heard_voice
            and elapsed >= self.min_seconds
            and now - self._last_voice_at >= self.silence_seconds
        )

    def stop(self) -> Path:
        if not self.recording:
            raise VoiceError("The microphone is not recording.")
        stream, self._stream = self._stream, None
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        with self._lock:
            frames = list(self._frames)
            self._frames = []
        if not frames:
            raise VoiceError("No audio was recorded. Please try again.")
        audio = np.concatenate(frames, axis=0).reshape(-1)
        duration = len(audio) / float(self.sample_rate)
        if duration < self.min_seconds:
            raise VoiceError("The recording was too short. Hold the microphone and speak again.")
        handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        path = Path(handle.name)
        handle.close()
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio.astype("<i2", copy=False).tobytes())
        return path

    def cancel(self) -> None:
        if self._stream is not None:
            stream, self._stream = self._stream, None
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        with self._lock:
            self._frames = []


_WHISPER_MODEL = None
_WHISPER_MODEL_KEY: tuple[str, str, str] | None = None
_WHISPER_LOCK = threading.Lock()

_ARABIC_TRANSCRIPTION_PROMPT = (
    "سؤال باللهجة المصرية عن جامعة المنصورة الجديدة، الكليات، البرامج، "
    "الأقسام، المصروفات، الرسوم، المنح، القبول، التحويل، السكن، المستشفى، "
    "مجلس الأمناء، رئيس الجامعة، عميد كلية الهندسة ومكان الجامعة."
)


def normalize_domain_transcript(text: str) -> str:
    """Repair a few repeatable Whisper merges in short Egyptian questions.

    Corrections are deliberately narrow and contextual; arbitrary user speech
    is otherwise returned untouched.  These mistakes were reproduced with a
    real Arabic TTS→Whisper round trip on the supported university questions.
    """
    value = re.sub(r"\s+", " ", (text or "").strip())
    value = re.sub(r"^(?:مرسوم|مارسوم)\b", "ما رسوم", value)
    value = re.sub(r"\bالطبل\b", "الطب", value)
    value = re.sub(r"\bقليات\b", "كليات", value)
    if "رسوم" in value:
        value = re.sub(r"\bكليات\s+(?=الطب|الهندسة|الصيدلة)", "كلية ", value)
    return value.strip()


def _get_whisper_model():
    global _WHISPER_MODEL, _WHISPER_MODEL_KEY
    key = (VOICE_STT_MODEL, VOICE_STT_DEVICE, VOICE_STT_COMPUTE_TYPE)
    with _WHISPER_LOCK:
        if _WHISPER_MODEL is None or _WHISPER_MODEL_KEY != key:
            try:
                from faster_whisper import WhisperModel
                from huggingface_hub import snapshot_download
                import truststore
            except ImportError as exc:
                raise VoiceError(
                    "Local speech recognition is not installed. Install faster-whisper first."
                ) from exc
            truststore.inject_into_ssl()
            model_source = VOICE_STT_MODEL
            # Once downloaded, resolve the cached snapshot locally so every
            # microphone click does not depend on Hugging Face being online.
            # If this is the first run, Faster Whisper keeps its normal
            # download behavior and obtains the model below.
            try:
                model_source = snapshot_download(
                    f"Systran/faster-whisper-{VOICE_STT_MODEL}",
                    local_files_only=True,
                )
            except Exception:
                pass
            _WHISPER_MODEL = WhisperModel(
                model_source,
                device=VOICE_STT_DEVICE,
                compute_type=VOICE_STT_COMPUTE_TYPE,
            )
            _WHISPER_MODEL_KEY = key
        return _WHISPER_MODEL


def _join_transcription_segments(segments) -> str:
    return " ".join(segment.text.strip() for segment in segments).strip()


def transcribe_audio(model, audio_path: Path) -> str:
    """Transcribe Arabic/English and recover from wrong language detection.

    Short Egyptian Arabic questions are occasionally classified as Turkish or
    another Latin-script language. Sending that transliteration to RAG changes
    the meaning completely. If auto-detection leaves the supported ar/en set
    (or reports very uncertain English), run one grounded Arabic pass before
    the text is allowed into `/chat`.
    """
    common_options = {
        "beam_size": 5,
        "vad_filter": True,
        "condition_on_previous_text": False,
    }
    segments, info = model.transcribe(str(audio_path), **common_options)
    text = _join_transcription_segments(segments)
    detected = str(getattr(info, "language", "") or "").lower()
    probability = float(getattr(info, "language_probability", 0.0) or 0.0)

    # Even a correctly detected Arabic pass benefits materially from an
    # explicit Arabic/domain pass (e.g. it fixes "قليات" to "كليات"). Only
    # confident English skips the second pass.
    retry_as_arabic = detected != "en" or probability < 0.60
    if retry_as_arabic:
        ar_segments, _ar_info = model.transcribe(
            str(audio_path),
            language="ar",
            initial_prompt=_ARABIC_TRANSCRIPTION_PROMPT,
            **common_options,
        )
        arabic_text = _join_transcription_segments(ar_segments)
        if arabic_text:
            return normalize_domain_transcript(arabic_text)
    return normalize_domain_transcript(text)


class TranscriptionWorker(QThread):
    succeeded = Signal(str)
    failed = Signal(str)

    def __init__(self, audio_path: Path) -> None:
        super().__init__()
        self.audio_path = audio_path

    def run(self) -> None:
        try:
            model = _get_whisper_model()
            text = transcribe_audio(model, self.audio_path)
            if not text:
                raise VoiceError("No clear speech was detected. Please try again.")
            self.succeeded.emit(text)
        except VoiceError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"Speech could not be transcribed: {exc}")
        finally:
            self.audio_path.unlink(missing_ok=True)


class SpeechWorker(QThread):
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = clean_text_for_speech(text)
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass

    def run(self) -> None:
        if not self.text or self._cancelled.is_set():
            self.finished_ok.emit()
            return
        cache_dir = Path(tempfile.gettempdir()) / "nmu_voice_cache"
        voice = (
            VOICE_TTS_ARABIC_VOICE
            if contains_arabic(self.text)
            else VOICE_TTS_ENGLISH_VOICE
        )
        key = hashlib.sha256(f"{voice}\0{self.text}".encode("utf-8")).hexdigest()
        audio_path = cache_dir / f"{key}.mp3"
        partial_path = cache_dir / f"{key}.part"
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            if not audio_path.exists() or audio_path.stat().st_size == 0:
                import edge_tts
                import truststore

                # Use Windows' trusted certificate store; never disable TLS
                # verification. This is needed on managed Windows networks.
                truststore.inject_into_ssl()
                partial_path.unlink(missing_ok=True)
                asyncio.run(
                    edge_tts.Communicate(self.text, voice=voice).save(
                        str(partial_path)
                    )
                )
                partial_path.replace(audio_path)
            if self._cancelled.is_set():
                self.finished_ok.emit()
                return

            import av
            import sounddevice as sd

            chunks: list[np.ndarray] = []
            sample_rate = 24000
            with av.open(str(audio_path)) as container:
                for frame in container.decode(audio=0):
                    sample_rate = frame.sample_rate or sample_rate
                    samples = frame.to_ndarray()
                    if samples.ndim > 1:
                        samples = samples.mean(axis=0)
                    chunks.append(np.asarray(samples).reshape(-1))
            if not chunks:
                raise VoiceError("The speech service returned empty audio.")
            audio = np.concatenate(chunks)
            if not self._cancelled.is_set():
                sd.play(audio, samplerate=sample_rate, blocking=True)
            self.finished_ok.emit()
        except Exception as exc:
            self.failed.emit(f"The answer could not be spoken: {exc}")
        finally:
            partial_path.unlink(missing_ok=True)

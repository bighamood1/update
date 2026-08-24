"""Pure unit tests for the desktop voice adapter (no mic/model downloads)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

GUI_DIR = Path(__file__).resolve().parents[1] / "gui"
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

from voice import (  # noqa: E402
    AudioRecorder,
    clean_text_for_speech,
    contains_arabic,
    normalize_domain_transcript,
    transcribe_audio,
)


def test_clean_text_for_speech_removes_markdown_and_urls():
    text = "## Programs\n- **Medicine**\n- [Engineering](https://nmu.edu.eg/x)"

    spoken = clean_text_for_speech(text)

    assert "Programs" in spoken
    assert "Medicine" in spoken
    assert "Engineering" in spoken
    assert "https://" not in spoken
    assert "**" not in spoken


def test_contains_arabic_detects_mixed_answer():
    assert contains_arabic("The location is في مدينة المنصورة الجديدة")
    assert not contains_arabic("New Mansoura University")


def test_recorder_auto_stops_after_speech_then_silence():
    now = [10.0]
    recorder = AudioRecorder(
        silence_seconds=1.0,
        min_seconds=0.5,
        max_seconds=20,
        clock=lambda: now[0],
    )
    recorder._stream = object()  # simulate an open stream without hardware
    recorder._started_at = 10.0
    recorder._heard_voice = True
    recorder._last_voice_at = 10.4

    now[0] = 11.0
    assert not recorder.should_auto_stop()
    now[0] = 11.5
    assert recorder.should_auto_stop()


def test_recorder_has_hard_maximum_duration():
    now = [5.0]
    recorder = AudioRecorder(max_seconds=3.0, clock=lambda: now[0])
    recorder._stream = object()
    recorder._started_at = 5.0

    now[0] = 8.1
    assert recorder.should_auto_stop()


def test_non_target_language_is_retranscribed_as_arabic(tmp_path):
    class FakeModel:
        def __init__(self):
            self.calls = []

        def transcribe(self, _path, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("language") == "ar":
                return (
                    [SimpleNamespace(text="ما هي كليات جامعة المنصورة الجديدة؟")],
                    SimpleNamespace(language="ar", language_probability=1.0),
                )
            return (
                [SimpleNamespace(text="Mahheye Kulliye cemaati munsura gidelim")],
                SimpleNamespace(language="tr", language_probability=0.91),
            )

    model = FakeModel()
    transcript = transcribe_audio(model, tmp_path / "voice.wav")

    assert transcript == "ما هي كليات جامعة المنصورة الجديدة؟"
    assert len(model.calls) == 2
    assert model.calls[1]["language"] == "ar"
    assert "جامعة المنصورة الجديدة" in model.calls[1]["initial_prompt"]


def test_confident_english_is_not_forced_to_arabic(tmp_path):
    class FakeModel:
        def transcribe(self, _path, **kwargs):
            assert "language" not in kwargs
            return (
                [SimpleNamespace(text="Where is New Mansoura University?")],
                SimpleNamespace(language="en", language_probability=0.98),
            )

    transcript = transcribe_audio(FakeModel(), tmp_path / "voice.wav")

    assert transcript == "Where is New Mansoura University?"


def test_detected_arabic_gets_domain_guided_second_pass(tmp_path):
    class FakeModel:
        def __init__(self):
            self.calls = 0

        def transcribe(self, _path, **kwargs):
            self.calls += 1
            if kwargs.get("language") == "ar":
                return (
                    [SimpleNamespace(text="ما هي كليات جامعة المنصورة الجديدة؟")],
                    SimpleNamespace(language="ar", language_probability=1.0),
                )
            return (
                [SimpleNamespace(text="ما هي قليات جامعة المنصورة الجديدة؟")],
                SimpleNamespace(language="ar", language_probability=0.99),
            )

    model = FakeModel()
    transcript = transcribe_audio(model, tmp_path / "voice.wav")

    assert transcript == "ما هي كليات جامعة المنصورة الجديدة؟"
    assert model.calls == 2


def test_short_arabic_fee_question_repairs_reproduced_whisper_merges():
    assert normalize_domain_transcript("مرسوم كليات الطبل") == "ما رسوم كلية الطب"
    assert normalize_domain_transcript("مارسوم كلية الطب") == "ما رسوم كلية الطب"

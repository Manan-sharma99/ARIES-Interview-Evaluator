import os
from functools import lru_cache

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")


@lru_cache(maxsize=1)
def get_whisper_model():
    """Load faster-whisper once and reuse it across Streamlit reruns."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Install it with: pip install faster-whisper"
        ) from exc

    return WhisperModel(
        WHISPER_MODEL_SIZE,
        device="cpu",
        compute_type="int8",
        cpu_threads=max(1, min(4, os.cpu_count() or 1)),
    )


def transcribe_audio(audio_path):
    """Transcribe an audio file using CPU-friendly faster-whisper settings."""
    if not audio_path or not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        segments, _ = get_whisper_model().transcribe(
            audio_path,
            language="en",
            beam_size=1,
            vad_filter=True,
        )
        transcript = " ".join(
            segment.text.strip()
            for segment in segments
            if segment.text and segment.text.strip()
        )
    except Exception as exc:
        raise RuntimeError(f"Whisper transcription failed: {exc}") from exc

    if not transcript:
        raise RuntimeError("Whisper could not detect any speech in the audio.")

    return transcript

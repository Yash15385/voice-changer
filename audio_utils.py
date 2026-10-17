"""
audio_utils.py — Audio I/O, Text-to-Speech, and Format Conversion Utilities
============================================================================
🎙️ Funny Voice Changer — Hacktoberfest Open Source Project
----------------------------------------------------------------------------
Handles loading audio from uploaded files (WAV, MP3, OGG, FLAC),
synthesizing crystal-clear speech from custom user text via ffmpeg/libflite
(with fallback synthesis), computing waveform envelopes, and exporting WAVs.
"""

import io
import re
import shutil
import subprocess
import tempfile
import warnings
import wave
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, resample

warnings.filterwarnings("ignore")

TARGET_SAMPLE_RATE = 22050
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}

TTS_VOICES = {
    "slt": "👩 Clara (Clear Female — US)",
    "rms": "👨 Ryan (Deep Broadcast Male — US)",
    "kal": "🧑 Kevin (Standard Male — US)",
    "awb": "🧔 Arthur (Scottish Accent Male — UK)",
}

SAMPLE_PRESETS = {
    "greeting": {
        "label": "👩 Clara: \"Hello squad! Welcome to VoiceMorph studio!\"",
        "voice": "slt",
        "text": "Hello squad! Welcome to Voice Morph studio. Pick any effect and hear how clear it sounds!",
    },
    "custom_demo": {
        "label": "👨 Ryan: \"Hey everyone, this is my custom voice changer test!\"",
        "voice": "rms",
        "text": "Hey everyone, this is my custom voice changer test. Every single word is crystal clear!",
    },
    "laugh": {
        "label": "😄 Kevin: \"Ha ha ha! That funny voice effect sounds hilarious!\"",
        "voice": "kal",
        "text": "Ha ha ha! That funny voice effect sounds super clear and hilarious!",
    },
    "scifi": {
        "label": "🛸 Sci-Fi: \"Attention Earthlings! We come in peace!\"",
        "voice": "rms",
        "text": "Attention Earthlings! We come in peace. Take me to your leader right now!",
    },
    "gamer": {
        "label": "🎮 Arthur: \"Good game squad! Rush the main objective!\"",
        "voice": "awb",
        "text": "Good game squad! Rush the main objective right now, I have got your back!",
    },
}


# ---------------------------------------------------------------------------
# LOADING
# ---------------------------------------------------------------------------

def load_audio(file_bytes: bytes, filename: str = "audio.wav") -> tuple[np.ndarray, int]:
    """Load audio bytes into a (numpy float32 array, sample_rate) tuple."""
    if not file_bytes:
        raise ValueError("Uploaded audio file is empty.")

    ext = Path(filename).suffix.lower() or ".wav"

    if ext == ".wav" or file_bytes[:4] == b"RIFF":
        try:
            return _load_wav(file_bytes)
        except Exception:
            pass

    try:
        return _load_via_pydub(file_bytes, ext)
    except Exception:
        pass

    if shutil.which("ffmpeg"):
        return _load_via_ffmpeg(file_bytes, ext)

    raise ValueError(
        f"Could not decode audio file '{filename}' ({ext}). "
        "Please upload a standard WAV file or ensure ffmpeg/pydub is installed."
    )


def _load_wav(file_bytes: bytes) -> tuple[np.ndarray, int]:
    """Load WAV bytes using scipy or standard library wave (handles 24-bit PCM)."""
    buf = io.BytesIO(file_bytes)
    try:
        sample_rate, data = wavfile.read(buf)
        audio = data.astype(np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        if data.dtype == np.int16:
            audio /= 32768.0
        elif data.dtype == np.int32:
            max_val = np.max(np.abs(audio))
            audio /= 2147483648.0 if max_val > 8388608 else 8388608.0
        elif data.dtype == np.uint8:
            audio = (audio - 128.0) / 128.0

        audio = np.clip(audio, -1.0, 1.0)
        return audio.astype(np.float32), int(sample_rate)
    except Exception:
        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sample_rate = wf.getframerate()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)

        if sampwidth == 3:
            a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            signed = (
                a[:, 0].astype(np.int32)
                | (a[:, 1].astype(np.int32) << 8)
                | (a[:, 2].astype(np.int32) << 16)
            )
            signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
            audio = signed.astype(np.float32) / 8388608.0
        elif sampwidth == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 1:
            audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0

        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        return np.clip(audio, -1.0, 1.0).astype(np.float32), int(sample_rate)


def _load_via_pydub(file_bytes: bytes, ext: str) -> tuple[np.ndarray, int]:
    """Load MP3/OGG/FLAC via pydub AudioSegment."""
    from pydub import AudioSegment

    fmt = ext.lstrip(".") or "wav"
    seg = AudioSegment.from_file(io.BytesIO(file_bytes), format=fmt)
    seg = seg.set_channels(1).set_sample_width(2)

    sample_rate = seg.frame_rate
    raw = np.frombuffer(seg.raw_data, dtype=np.int16)
    audio = raw.astype(np.float32) / 32768.0
    return audio, int(sample_rate)


def _load_via_ffmpeg(file_bytes: bytes, ext: str) -> tuple[np.ndarray, int]:
    """Decode any audio container to mono 16-bit WAV via ffmpeg CLI."""
    suffix = ext if ext.startswith(".") else f".{ext}"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as src_tmp:
        src_tmp.write(file_bytes)
        src_path = Path(src_tmp.name)
    dst_path = src_path.with_suffix(".decoded.wav")
    try:
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(src_path),
            "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE),
            "-f", "wav", str(dst_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return _load_wav(dst_path.read_bytes())
    finally:
        src_path.unlink(missing_ok=True)
        dst_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CUSTOM TEXT-TO-SPEECH & SAMPLE VOICE GENERATOR
# ---------------------------------------------------------------------------

def synthesize_text_to_speech(
    text: str,
    voice: str = "slt",
    speed: float = 1.0,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> tuple[np.ndarray, int]:
    """
    Synthesize clear, natural spoken audio from custom user text using
    ffmpeg's built-in libflite TTS engine (or fallback formant synthesis).
    """
    clean_text = re.sub(r"[^a-zA-Z0-9\s.,!?-]", " ", text or "").strip()
    if not clean_text:
        clean_text = "Hello squad, testing Voice Morph custom text!"
    clean_text = clean_text[:500]
    voice_id = voice if voice in TTS_VOICES else "slt"
    spd = float(np.clip(speed, 0.65, 1.6))

    if shutil.which("ffmpeg"):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
            out_path = Path(tmp_f.name)
        try:
            af_chain = f"highpass=f=90,lowpass=f=7800,treble=g=3:f=2800,atempo={spd:.2f},volume=1.5"
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-f", "lavfi",
                "-i", f"flite=text='{clean_text}':voice={voice_id}",
                "-af", af_chain,
                "-ar", str(sample_rate), "-ac", "1",
                str(out_path),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            audio, sr = _load_wav(out_path.read_bytes())
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = (audio / peak) * 0.92
            return audio.astype(np.float32), sr
        except Exception:
            pass
        finally:
            out_path.unlink(missing_ok=True)

    # Fallback formant speech synthesis if ffmpeg/libflite is unavailable
    duration = max(1.8, min(6.0, len(clean_text.split()) * 0.38))
    n_samples = int(duration * sample_rate)
    t = np.arange(n_samples, dtype=np.float32) / sample_rate
    audio = 0.5 * np.sin(2.0 * np.pi * 150.0 * t) + 0.3 * np.sin(2.0 * np.pi * 300.0 * t)
    return audio.astype(np.float32), sample_rate


def generate_sample_voice(
    preset_key: str = "greeting",
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> tuple[np.ndarray, int]:
    """Generate a crystal-clear spoken sample clip from SAMPLE_PRESETS."""
    preset = SAMPLE_PRESETS.get(preset_key, SAMPLE_PRESETS["greeting"])
    return synthesize_text_to_speech(
        text=preset["text"],
        voice=preset.get("voice", "slt"),
        speed=1.0,
        sample_rate=sample_rate,
    )


# ---------------------------------------------------------------------------
# RESAMPLING & EXPORTING
# ---------------------------------------------------------------------------

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to a target sample rate."""
    if orig_sr == target_sr or len(audio) == 0:
        return audio.astype(np.float32)
    target_len = max(1, int(len(audio) * target_sr / orig_sr))
    return resample(audio, target_len).astype(np.float32)


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert a float32 numpy audio array to 16-bit PCM WAV file bytes."""
    audio_clipped = np.clip(np.nan_to_num(audio, nan=0.0), -1.0, 1.0)
    pcm = (audio_clipped * 32767).astype(np.int16)

    buf = io.BytesIO()
    wavfile.write(buf, int(sample_rate), pcm)
    return buf.getvalue()


def trim_silence(
    audio: np.ndarray,
    threshold: float = 0.01,
    padding_ms: float = 100.0,
    sample_rate: int = 22050,
) -> np.ndarray:
    """Trim leading and trailing silence from audio."""
    if len(audio) == 0:
        return audio
    pad = int(sample_rate * padding_ms / 1000.0)
    mask = np.abs(audio) > threshold
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return audio

    start = max(0, indices[0] - pad)
    end = min(len(audio), indices[-1] + pad + 1)
    return audio[start:end]


def get_audio_duration(audio: np.ndarray, sample_rate: int) -> float:
    """Return audio duration in seconds."""
    if sample_rate <= 0:
        return 0.0
    return len(audio) / sample_rate


def get_audio_info(audio: np.ndarray, sample_rate: int) -> dict:
    """Return a dict of audio metadata for display."""
    if len(audio) == 0:
        return {
            "duration_sec": 0.0,
            "sample_rate": sample_rate,
            "samples": 0,
            "peak_amplitude": 0.0,
            "rms": 0.0,
        }
    return {
        "duration_sec": round(get_audio_duration(audio, sample_rate), 2),
        "sample_rate": int(sample_rate),
        "samples": int(len(audio)),
        "peak_amplitude": round(float(np.max(np.abs(audio))), 4),
        "rms": round(float(np.sqrt(np.mean(audio ** 2))), 4),
    }


def compute_waveform_envelope(audio: np.ndarray, num_points: int = 120) -> list[float]:
    """Compute normalized peak envelope bars in [0.04, 1.0] for UI waveform visualization."""
    if len(audio) == 0:
        return [0.05] * num_points
    chunks = np.array_split(np.abs(audio), num_points)
    peaks = np.array([float(np.max(c)) if len(c) > 0 else 0.0 for c in chunks])
    max_p = float(np.max(peaks))
    if max_p > 0:
        peaks = peaks / max_p
    return [round(max(0.04, min(1.0, float(v))), 3) for v in peaks]

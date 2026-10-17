"""
effects.py — High-Clarity DSP (Digital Signal Processing) Audio Effects Engine
==============================================================================
🎙️ Funny Voice Changer — Hacktoberfest Open Source Project
------------------------------------------------------------------------------
Engineered for maximum speech intelligibility:
  - Phase-aligned Cross-Correlated SOLA (Synchronized Overlap-Add) pitch shifting
    eliminates hollow comb-filtering artifacts.
  - Consonant clarity preservation (1.5–4.5 kHz presence band) keeps spoken
    words crisp and easy to understand across all 12 voice effects.
"""

import warnings
import numpy as np
from scipy.signal import butter, lfilter, resample

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# UTILITY & VOCAL CLARITY HELPERS
# ---------------------------------------------------------------------------

def normalize(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-0.96, 0.96] range. Handles silence gracefully."""
    if len(audio) == 0:
        return audio.astype(np.float32)
    peak = np.max(np.abs(audio))
    if peak == 0 or not np.isfinite(peak):
        return np.zeros_like(audio, dtype=np.float32)
    return ((audio / peak) * 0.96).astype(np.float32)


def ensure_float32(audio: np.ndarray) -> np.ndarray:
    """Convert any integer audio array to float32 in [-1, 1]."""
    if audio.dtype == np.float32 or audio.dtype == np.float64:
        return audio.astype(np.float32)
    if audio.dtype == np.int16:
        return (audio / 32768.0).astype(np.float32)
    if audio.dtype == np.int32:
        return (audio / 2147483648.0).astype(np.float32)
    if audio.dtype == np.uint8:
        return ((audio.astype(np.float32) - 128.0) / 128.0).astype(np.float32)
    return audio.astype(np.float32)


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert stereo/multi-channel audio to mono by averaging channels."""
    if audio.ndim == 2:
        return audio.mean(axis=1).astype(np.float32)
    return audio.astype(np.float32)


def butter_lowpass(cutoff: float, sample_rate: int, order: int = 3):
    nyq = 0.5 * sample_rate
    normal_cutoff = min(max(cutoff / nyq, 0.01), 0.98)
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_highpass(cutoff: float, sample_rate: int, order: int = 3):
    nyq = 0.5 * sample_rate
    normal_cutoff = min(max(cutoff / nyq, 0.01), 0.98)
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return b, a


def butter_bandpass(lowcut: float, highcut: float, sample_rate: int, order: int = 3):
    nyq = 0.5 * sample_rate
    low = min(max(lowcut / nyq, 0.01), 0.94)
    high = min(max(highcut / nyq, low + 0.02), 0.98)
    b, a = butter(order, [low, high], btype="band", analog=False)
    return b, a


def enhance_vocal_clarity(
    wet_audio: np.ndarray,
    dry_audio: np.ndarray,
    sample_rate: int,
    clarity_boost: float = 0.32,
) -> np.ndarray:
    """
    Boost speech intelligibility by reinforcing the 1.4–4.2 kHz vocal presence
    and consonant articulation band so effects never sound muddy.
    """
    b_bp, a_bp = butter_bandpass(1400.0, 4200.0, sample_rate, order=2)
    wet_presence = lfilter(b_bp, a_bp, wet_audio).astype(np.float32)
    boosted = wet_audio + float(clarity_boost) * wet_presence

    # Optional subtle dry consonant articulation blend when lengths match
    if len(wet_audio) == len(dry_audio) and clarity_boost > 0:
        dry_consonants = lfilter(b_bp, a_bp, dry_audio).astype(np.float32)
        boosted = boosted + (float(clarity_boost) * 0.25) * dry_consonants

    return normalize(boosted)


def pitch_shift_resample(audio: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
    """
    Phase-Aligned Cross-Correlated SOLA (Synchronized Overlap-Add) Pitch Shifter:
      - Searches a local cross-correlation window for each overlapping grain so
        pitch cycles align in phase, preventing hollow comb-filtering artifacts
        and keeping spoken words crisp and intelligible.
    """
    if abs(semitones) < 0.05 or len(audio) < 128:
        return audio.astype(np.float32)

    factor = 2.0 ** (semitones / 12.0)
    original_length = len(audio)

    scaled_len = max(64, int(np.round(original_length / factor)))
    pitched = resample(audio, scaled_len).astype(np.float32)

    win_size = int(0.030 * sample_rate)  # 30 ms grain window
    win_size = max(256, min(1024, win_size - (win_size % 2)))
    hop_out = win_size // 2
    max_search = min(win_size // 4, int(0.007 * sample_rate))

    fade_in = np.linspace(0.0, 1.0, hop_out, dtype=np.float32)
    fade_out = 1.0 - fade_in

    output = np.zeros(original_length + win_size, dtype=np.float32)
    first_grain_len = min(win_size, len(pitched))
    output[:first_grain_len] = pitched[:first_grain_len]

    out_pos = hop_out
    while out_pos + win_size <= original_length:
        expected_in = int(np.round((out_pos / original_length) * (len(pitched) - win_size)))
        expected_in = max(0, min(len(pitched) - win_size, expected_in))

        # Cross-correlation search to align fundamental pitch periods
        s_start = max(0, expected_in - max_search)
        s_end = min(len(pitched) - win_size, expected_in + max_search)
        tail_ref = output[out_pos : out_pos + hop_out]

        best_idx = expected_in
        if s_end > s_start and np.max(np.abs(tail_ref)) > 1e-4:
            best_corr = -1e9
            step = 2
            for cand in range(s_start, s_end + 1, step):
                cand_seg = pitched[cand : cand + hop_out]
                corr = float(np.dot(tail_ref, cand_seg))
                if corr > best_corr:
                    best_corr = corr
                    best_idx = cand

        grain = pitched[best_idx : best_idx + win_size]
        # Crossfade overlap region cleanly
        output[out_pos : out_pos + hop_out] = (
            output[out_pos : out_pos + hop_out] * fade_out + grain[:hop_out] * fade_in
        )
        output[out_pos + hop_out : out_pos + win_size] = grain[hop_out:]
        out_pos += hop_out

    return normalize(output[:original_length])


# ---------------------------------------------------------------------------
# BUILT-IN HIGH-CLARITY EFFECTS
# ---------------------------------------------------------------------------

def effect_helium(
    audio: np.ndarray,
    sample_rate: int,
    semitones: float = 4.5,
    clarity: float = 0.45,
) -> np.ndarray:
    """
    🎈 HELIUM — Clean +4.5 semitone pitch shift up with crisp vocal articulation.
    """
    shifted = pitch_shift_resample(audio, sample_rate, semitones=float(semitones))
    b, a = butter_highpass(110.0, sample_rate, order=2)
    filtered = lfilter(b, a, shifted).astype(np.float32)
    return enhance_vocal_clarity(filtered, shifted, sample_rate, clarity_boost=float(clarity))


def effect_monster(
    audio: np.ndarray,
    sample_rate: int,
    semitones: float = 4.0,
    growl_drive: float = 1.6,
    clarity: float = 0.55,
) -> np.ndarray:
    """
    👹 MONSTER — Deep -4.0 semitone pitch drop + warm bass growl while keeping
    high-frequency vocal consonants crystal clear.
    """
    shifted = pitch_shift_resample(audio, sample_rate, semitones=-abs(float(semitones)))
    b_lp, a_lp = butter_lowpass(650.0, sample_rate, order=2)
    b_hp, a_hp = butter_highpass(650.0, sample_rate, order=2)

    low_band = lfilter(b_lp, a_lp, shifted).astype(np.float32)
    high_band = lfilter(b_hp, a_hp, shifted).astype(np.float32)

    driven_low = np.tanh(low_band * float(growl_drive)) * 0.9
    blended = driven_low * 0.65 + high_band * 0.85
    return enhance_vocal_clarity(blended, shifted, sample_rate, clarity_boost=float(clarity))


def effect_robotic(
    audio: np.ndarray,
    sample_rate: int,
    carrier_freq: float = 55.0,
    modulation_depth: float = 0.55,
    clarity: float = 0.50,
) -> np.ndarray:
    """
    🤖 ROBOTIC — 55 Hz sci-fi droid ring modulation + 12 ms metallic comb
    resonator blended with crisp speech consonants for 100% intelligibility.
    """
    t = np.arange(len(audio), dtype=np.float32) / sample_rate
    carrier = (
        0.8 * np.sin(2.0 * np.pi * float(carrier_freq) * t)
        + 0.2 * np.cos(4.0 * np.pi * float(carrier_freq) * t)
    ).astype(np.float32)

    depth = float(np.clip(modulation_depth, 0.1, 0.9))
    ring_mod = audio * (depth * carrier + (1.0 - depth))

    # Add short 12 ms metallic droid resonator
    comb_delay = max(1, int(0.012 * sample_rate))
    comb = ring_mod.copy()
    if comb_delay < len(comb):
        comb[comb_delay:] += ring_mod[:-comb_delay] * 0.38

    b_hp, a_hp = butter_highpass(160.0, sample_rate, order=2)
    result = lfilter(b_hp, a_hp, comb).astype(np.float32)
    return enhance_vocal_clarity(result, audio, sample_rate, clarity_boost=float(clarity))


def effect_echo(
    audio: np.ndarray,
    sample_rate: int,
    delay_ms: float = 240.0,
    decay: float = 0.38,
    num_echoes: int = 3,
) -> np.ndarray:
    """
    🔊 ECHO — Crystal-clear upfront dry voice followed by distinct multi-tap echoes.
    """
    delay_samples = max(1, int(sample_rate * float(delay_ms) / 1000.0))
    n_echoes = max(1, int(num_echoes))
    total_len = len(audio) + delay_samples * n_echoes
    result = np.zeros(total_len, dtype=np.float32)
    result[: len(audio)] = audio

    # High-pass echoes slightly so repeats don't muddy the main voice
    b_hp, a_hp = butter_highpass(260.0, sample_rate, order=2)
    clean_echo_src = lfilter(b_hp, a_hp, audio).astype(np.float32)

    for i in range(1, n_echoes + 1):
        shift = i * delay_samples
        gain = float(decay) ** i
        result[shift : shift + len(audio)] += clean_echo_src * gain

    return normalize(result)


def effect_chipmunk(
    audio: np.ndarray,
    sample_rate: int,
    semitones: float = 6.5,
    speed_factor: float = 1.15,
) -> np.ndarray:
    """
    🐿️ CHIPMUNK — Bright +6.5 semitone cartoon pitch shift with gentle 1.15x speed
    so every word remains clear and funny.
    """
    shifted = pitch_shift_resample(audio, sample_rate, semitones=float(semitones))
    spd = float(np.clip(speed_factor, 0.9, 1.5))
    if abs(spd - 1.0) > 0.02 and len(shifted) > 32:
        new_len = max(16, int(len(shifted) / spd))
        shifted = resample(shifted, new_len).astype(np.float32)
    b, a = butter_highpass(180.0, sample_rate, order=2)
    result = lfilter(b, a, shifted).astype(np.float32)
    return enhance_vocal_clarity(result, result, sample_rate, clarity_boost=0.35)


def effect_cave(
    audio: np.ndarray,
    sample_rate: int,
    room_decay: float = 0.50,
    wet_mix: float = 0.35,
) -> np.ndarray:
    """
    🏔️ CAVE — Clear upfront voice with a 55 ms pre-delay and spacious cavern tail.
    """
    delays_ms = [55, 79, 107, 139]
    decay = float(np.clip(room_decay, 0.2, 0.8))
    mix = float(np.clip(wet_mix, 0.1, 0.7))
    tail_samples = int(sample_rate * 0.9)

    wet = np.zeros(len(audio) + tail_samples, dtype=np.float32)
    for d_ms in delays_ms:
        d_samples = max(1, int(sample_rate * d_ms / 1000.0))
        max_taps = min(10, int(tail_samples / d_samples) + 1)
        for i in range(1, max_taps):
            gain = (decay ** i) * 0.30
            start = i * d_samples
            end = start + len(audio)
            if end <= len(wet):
                wet[start:end] += audio * gain

    b_lp, a_lp = butter_lowpass(3200.0, sample_rate, order=2)
    b_hp, a_hp = butter_highpass(220.0, sample_rate, order=2)
    wet = lfilter(b_hp, a_hp, lfilter(b_lp, a_lp, wet)).astype(np.float32)

    result = wet * mix
    result[: len(audio)] += audio * (1.0 - mix * 0.25)
    return normalize(result)


def effect_tremolo(
    audio: np.ndarray,
    sample_rate: int,
    rate: float = 5.5,
    depth: float = 0.60,
) -> np.ndarray:
    """
    🌊 TREMOLO — Smooth LFO amplitude modulation that keeps syllables audible.
    """
    t = np.arange(len(audio), dtype=np.float32) / sample_rate
    d = float(np.clip(depth, 0.1, 0.85))
    lfo = (1.0 - d * 0.45) + (d * 0.45) * np.sin(2.0 * np.pi * float(rate) * t)
    return enhance_vocal_clarity(audio * lfo, audio, sample_rate, clarity_boost=0.25)


def effect_vibrato(
    audio: np.ndarray,
    sample_rate: int,
    rate: float = 5.5,
    width_ms: float = 1.8,
) -> np.ndarray:
    """
    🎸 VIBRATO — Subtle sinusoidal pitch warble that preserves word clarity.
    """
    n = len(audio)
    if n < 16:
        return audio.astype(np.float32)
    t = np.arange(n, dtype=np.float32) / sample_rate
    max_delay = (float(width_ms) / 1000.0) * sample_rate
    delay = max_delay * (0.5 + 0.5 * np.sin(2.0 * np.pi * float(rate) * t))
    read_idx = np.clip(np.arange(n, dtype=np.float32) - delay, 0, n - 1)
    result = np.interp(read_idx, np.arange(n, dtype=np.float32), audio).astype(np.float32)
    return enhance_vocal_clarity(result, audio, sample_rate, clarity_boost=0.25)


def effect_telephone(
    audio: np.ndarray,
    sample_rate: int,
    lowcut: float = 320.0,
    highcut: float = 3400.0,
    grit: float = 1.25,
) -> np.ndarray:
    """
    📞 TELEPHONE — Crisp 320–3400 Hz telecom bandpass filter with warm presence.
    """
    b, a = butter_bandpass(float(lowcut), float(highcut), sample_rate, order=3)
    filtered = lfilter(b, a, audio).astype(np.float32)
    saturated = np.tanh(filtered * float(grit))
    return normalize(saturated)


def effect_underwater(
    audio: np.ndarray,
    sample_rate: int,
    cutoff_hz: float = 950.0,
    wobble_rate: float = 2.5,
    clarity: float = 0.40,
) -> np.ndarray:
    """
    🫧 UNDERWATER — Bubbly submerged filter blended with vocal mid-band clarity
    so you can still understand every spoken word underwater.
    """
    b_lp, a_lp = butter_lowpass(float(cutoff_hz), sample_rate, order=3)
    muffled = lfilter(b_lp, a_lp, audio).astype(np.float32)
    wobbled = effect_vibrato(muffled, sample_rate, rate=float(wobble_rate), width_ms=1.6)
    t = np.arange(len(wobbled), dtype=np.float32) / sample_rate
    bubble_env = 0.80 + 0.20 * np.sin(2.0 * np.pi * (float(wobble_rate) * 1.8) * t)
    submerged = wobbled * bubble_env
    return enhance_vocal_clarity(submerged, audio, sample_rate, clarity_boost=float(clarity))


def effect_chorus(
    audio: np.ndarray,
    sample_rate: int,
    voices: int = 3,
    depth_ms: float = 3.5,
    wet_mix: float = 0.45,
) -> np.ndarray:
    """
    🎶 CHORUS — Tight multi-voice ensemble with crisp lead vocal upfront.
    """
    n = len(audio)
    if n < 16:
        return audio.astype(np.float32)
    t = np.arange(n, dtype=np.float32) / sample_rate
    base_indices = np.arange(n, dtype=np.float32)
    wet = np.zeros(n, dtype=np.float32)
    num_voices = max(1, min(5, int(voices)))

    for v in range(num_voices):
        rate = 0.8 + 0.35 * v
        phase = (2.0 * np.pi * v) / num_voices
        center_delay = ((7.0 + 3.5 * v) / 1000.0) * sample_rate
        mod_depth = (float(depth_ms) / 1000.0) * sample_rate
        delay = center_delay + mod_depth * np.sin(2.0 * np.pi * rate * t + phase)
        read_idx = np.clip(base_indices - delay, 0, n - 1)
        wet += np.interp(read_idx, base_indices, audio).astype(np.float32)

    wet /= num_voices
    mix = float(np.clip(wet_mix, 0.1, 0.8))
    blended = (1.0 - mix * 0.5) * audio + mix * wet
    return enhance_vocal_clarity(blended, audio, sample_rate, clarity_boost=0.30)


def effect_flanger(
    audio: np.ndarray,
    sample_rate: int,
    rate: float = 0.5,
    max_delay_ms: float = 3.2,
    mix: float = 0.45,
) -> np.ndarray:
    """
    🌀 FLANGER — Smooth sci-fi jet-sweep comb delay with clear lead vocal.
    """
    n = len(audio)
    if n < 16:
        return audio.astype(np.float32)
    t = np.arange(n, dtype=np.float32) / sample_rate
    base_indices = np.arange(n, dtype=np.float32)
    max_delay = (float(max_delay_ms) / 1000.0) * sample_rate
    delay = 0.8 + max_delay * (0.5 + 0.5 * np.sin(2.0 * np.pi * float(rate) * t))
    read_idx = np.clip(base_indices - delay, 0, n - 1)
    delayed = np.interp(read_idx, base_indices, audio).astype(np.float32)
    m = float(np.clip(mix, 0.1, 0.85))
    blended = (1.0 - m * 0.35) * audio + (m * 0.65) * delayed
    return enhance_vocal_clarity(blended, audio, sample_rate, clarity_boost=0.30)


# ---------------------------------------------------------------------------
# EFFECTS REGISTRY
# ---------------------------------------------------------------------------

EFFECTS_REGISTRY: dict[str, dict] = {
    "helium": {
        "fn": effect_helium,
        "label": "🎈 Helium",
        "description": "Crisp pitch shift up — funny helium voice with clear words!",
        "params": {"semitones": 4.5, "clarity": 0.45},
        "param_schema": {
            "semitones": {"label": "Pitch Shift (Semitones)", "min": 2.0, "max": 9.0, "step": 0.5, "default": 4.5},
            "clarity": {"label": "Vocal Clarity Boost", "min": 0.0, "max": 0.8, "step": 0.05, "default": 0.45},
        },
        "color": "#f59e0b",
        "emoji": "🎈",
    },
    "monster": {
        "fn": effect_monster,
        "label": "👹 Monster",
        "description": "Deep pitch drop + bass growl with crisp consonant clarity.",
        "params": {"semitones": 4.0, "growl_drive": 1.6, "clarity": 0.55},
        "param_schema": {
            "semitones": {"label": "Pitch Drop (Semitones)", "min": 2.0, "max": 8.0, "step": 0.5, "default": 4.0},
            "growl_drive": {"label": "Bass Growl Drive", "min": 1.0, "max": 3.0, "step": 0.2, "default": 1.6},
            "clarity": {"label": "Vocal Clarity Boost", "min": 0.1, "max": 0.9, "step": 0.05, "default": 0.55},
        },
        "color": "#ef4444",
        "emoji": "👹",
    },
    "robotic": {
        "fn": effect_robotic,
        "label": "🤖 Robotic",
        "description": "Sci-fi droid ring modulation + metallic resonator with clear speech.",
        "params": {"carrier_freq": 55.0, "modulation_depth": 0.55, "clarity": 0.50},
        "param_schema": {
            "carrier_freq": {"label": "Droid Carrier (Hz)", "min": 30.0, "max": 120.0, "step": 5.0, "default": 55.0},
            "modulation_depth": {"label": "Robotic Depth", "min": 0.2, "max": 0.85, "step": 0.05, "default": 0.55},
            "clarity": {"label": "Vocal Clarity Boost", "min": 0.1, "max": 0.9, "step": 0.05, "default": 0.50},
        },
        "color": "#06b6d4",
        "emoji": "🤖",
    },
    "echo": {
        "fn": effect_echo,
        "label": "🔊 Echo",
        "description": "Clear upfront voice followed by clean canyon delay repeats.",
        "params": {"delay_ms": 240.0, "decay": 0.38, "num_echoes": 3},
        "param_schema": {
            "delay_ms": {"label": "Delay Time (ms)", "min": 80.0, "max": 500.0, "step": 20.0, "default": 240.0},
            "decay": {"label": "Echo Decay", "min": 0.15, "max": 0.70, "step": 0.05, "default": 0.38},
            "num_echoes": {"label": "Number of Echoes", "min": 1, "max": 6, "step": 1, "default": 3},
        },
        "color": "#8b5cf6",
        "emoji": "🔊",
    },
    "chipmunk": {
        "fn": effect_chipmunk,
        "label": "🐿️ Chipmunk",
        "description": "High cartoon squeak + lively speed while keeping words clear!",
        "params": {"semitones": 6.5, "speed_factor": 1.15},
        "param_schema": {
            "semitones": {"label": "Pitch Shift (Semitones)", "min": 4.0, "max": 10.0, "step": 0.5, "default": 6.5},
            "speed_factor": {"label": "Speed Multiplier", "min": 1.0, "max": 1.35, "step": 0.05, "default": 1.15},
        },
        "color": "#f97316",
        "emoji": "🐿️",
    },
    "cave": {
        "fn": effect_cave,
        "label": "🏔️ Cave",
        "description": "Pre-delayed Schroeder cavern reverb that keeps your voice upfront.",
        "params": {"room_decay": 0.50, "wet_mix": 0.35},
        "param_schema": {
            "room_decay": {"label": "Cavern Decay", "min": 0.25, "max": 0.75, "step": 0.05, "default": 0.50},
            "wet_mix": {"label": "Reverb Mix", "min": 0.15, "max": 0.65, "step": 0.05, "default": 0.35},
        },
        "color": "#64748b",
        "emoji": "🏔️",
    },
    "tremolo": {
        "fn": effect_tremolo,
        "label": "🌊 Tremolo",
        "description": "Smooth rhythmic volume wave modulation.",
        "params": {"rate": 5.5, "depth": 0.60},
        "param_schema": {
            "rate": {"label": "Wobble Rate (Hz)", "min": 2.0, "max": 12.0, "step": 0.5, "default": 5.5},
            "depth": {"label": "Modulation Depth", "min": 0.2, "max": 0.85, "step": 0.05, "default": 0.60},
        },
        "color": "#22c55e",
        "emoji": "🌊",
    },
    "vibrato": {
        "fn": effect_vibrato,
        "label": "🎸 Vibrato",
        "description": "Expressive musical pitch warble with crisp articulation.",
        "params": {"rate": 5.5, "width_ms": 1.8},
        "param_schema": {
            "rate": {"label": "Vibrato Speed (Hz)", "min": 2.0, "max": 10.0, "step": 0.5, "default": 5.5},
            "width_ms": {"label": "Pitch Width (ms)", "min": 0.5, "max": 4.0, "step": 0.2, "default": 1.8},
        },
        "color": "#ec4899",
        "emoji": "🎸",
    },
    "telephone": {
        "fn": effect_telephone,
        "label": "📞 Telephone",
        "description": "Authentic 320–3400 Hz phone line filter with clear voice presence.",
        "params": {"lowcut": 320.0, "highcut": 3400.0, "grit": 1.25},
        "param_schema": {
            "lowcut": {"label": "Low Cut (Hz)", "min": 200.0, "max": 600.0, "step": 40.0, "default": 320.0},
            "highcut": {"label": "High Cut (Hz)", "min": 2400.0, "max": 4500.0, "step": 100.0, "default": 3400.0},
            "grit": {"label": "Line Warmth", "min": 1.0, "max": 2.4, "step": 0.1, "default": 1.25},
        },
        "color": "#eab308",
        "emoji": "📞",
    },
    "underwater": {
        "fn": effect_underwater,
        "label": "🫧 Underwater",
        "description": "Submerged bubbly filter with vocal clarity blend so words stay clear.",
        "params": {"cutoff_hz": 950.0, "wobble_rate": 2.5, "clarity": 0.40},
        "param_schema": {
            "cutoff_hz": {"label": "Submerge Filter (Hz)", "min": 500.0, "max": 1800.0, "step": 50.0, "default": 950.0},
            "wobble_rate": {"label": "Bubble Speed (Hz)", "min": 1.0, "max": 5.0, "step": 0.2, "default": 2.5},
            "clarity": {"label": "Vocal Clarity Boost", "min": 0.1, "max": 0.8, "step": 0.05, "default": 0.40},
        },
        "color": "#38bdf8",
        "emoji": "🫧",
    },
    "chorus": {
        "fn": effect_chorus,
        "label": "🎶 Chorus",
        "description": "Multi-voice detuned ensemble with a clear lead voice.",
        "params": {"voices": 3, "depth_ms": 3.5, "wet_mix": 0.45},
        "param_schema": {
            "voices": {"label": "Ensemble Voices", "min": 2, "max": 5, "step": 1, "default": 3},
            "depth_ms": {"label": "Detune Depth (ms)", "min": 1.0, "max": 7.0, "step": 0.5, "default": 3.5},
            "wet_mix": {"label": "Chorus Mix", "min": 0.15, "max": 0.75, "step": 0.05, "default": 0.45},
        },
        "color": "#a855f7",
        "emoji": "🎶",
    },
    "flanger": {
        "fn": effect_flanger,
        "label": "🌀 Flanger",
        "description": "Swept sci-fi comb filter with crisp vocal intelligibility.",
        "params": {"rate": 0.5, "max_delay_ms": 3.2, "mix": 0.45},
        "param_schema": {
            "rate": {"label": "Sweep Rate (Hz)", "min": 0.1, "max": 2.5, "step": 0.1, "default": 0.5},
            "max_delay_ms": {"label": "Sweep Depth (ms)", "min": 1.0, "max": 6.0, "step": 0.4, "default": 3.2},
            "mix": {"label": "Flange Mix", "min": 0.15, "max": 0.75, "step": 0.05, "default": 0.45},
        },
        "color": "#14b8a6",
        "emoji": "🌀",
    },
}


def apply_effect(
    effect_key: str,
    audio: np.ndarray,
    sample_rate: int,
    **override_params,
) -> np.ndarray:
    """Apply a registered effect to an audio array."""
    if effect_key not in EFFECTS_REGISTRY:
        raise ValueError(
            f"Unknown effect '{effect_key}'. "
            f"Available: {list(EFFECTS_REGISTRY.keys())}"
        )

    entry = EFFECTS_REGISTRY[effect_key]
    fn = entry["fn"]
    params = dict(entry.get("params", {}))
    for k, v in override_params.items():
        if v is not None and k in params:
            params[k] = v

    audio = ensure_float32(audio)
    audio = to_mono(audio)
    return fn(audio, sample_rate, **params)

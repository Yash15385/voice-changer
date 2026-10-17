# 🎙️ VoiceMorph — Funny Voice Changer & Group Audio Sharing Messenger

[![Hacktoberfest](https://img.shields.io/badge/Hacktoberfest-2024-orange?logo=hacktoberfest&style=flat-square)](https://hacktoberfest.com)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&style=flat-square)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit&style=flat-square)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square)](CONTRIBUTING.md)

> A 100% free, open-source Python web application to apply funny voice effects to audio clips and share them with friend groups — built for Hacktoberfest contributors!

---

## ✨ Features

| Feature | Status |
|---|---|
| 🎈 Helium pitch-up effect | ✅ Done |
| 👹 Monster pitch-down + growl | ✅ Done |
| 🤖 Robotic ring modulation | ✅ Done |
| 🔊 Echo / delay | ✅ Done |
| 🐿️ Chipmunk (octave up) | ✅ Done |
| 🏔️ Cave reverb simulation | ✅ Done |
| 📁 WAV / MP3 upload | ✅ Done |
| 🎤 Browser microphone recording | ✅ Done |
| 👥 Friend group feeds (JSON) | ✅ Done |
| 📤 Share clips to group feed | ✅ Done |
| 📡 Live call room scaffold | 🚧 Contributor Opportunity |
| 🌊 Tremolo / Vibrato effect | 🆕 Good First Issue |
| 📞 WebSocket streaming | 🆕 Contributor Opportunity |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- (Optional) `ffmpeg` on your PATH for MP3 support

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-org/voicemorph
cd voicemorph

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run main.py
```

Open your browser at **http://localhost:8501** 🎉

---

## 📁 Project Structure

```
voicemorph/
├── main.py          ← Streamlit app — all pages & routing
├── effects.py       ← 🎛️ DSP effects engine (add yours here!)
├── audio_utils.py   ← Audio loading, resampling, WAV export
├── storage.py       ← JSON file-based state (no DB needed)
├── live_call.py     ← 📡 WebSocket/live call scaffold
├── requirements.txt ← Python dependencies
├── README.md        ← This file
├── CONTRIBUTING.md  ← Contribution guidelines
├── .streamlit/
│   └── config.toml  ← Streamlit theme config
└── data/            ← Auto-created at runtime
    ├── groups.json  ← Group definitions
    ├── clips/       ← Processed audio WAV files
    └── feed/        ← Per-group JSON feed files
```

---

## 🎛️ Adding a Custom Voice Effect (Hacktoberfest!)

Adding your own effect is just **3 steps** — all in `effects.py`:

### Step 1: Write the function

```python
import numpy as np
from effects import normalize  # already imported at top of effects.py

def effect_tremolo(audio: np.ndarray, sample_rate: int, rate: float = 5.0) -> np.ndarray:
    """
    🌊 TREMOLO — LFO amplitude modulation.

    Algorithm:
        Multiplies the signal by a slow sine LFO, creating a
        rhythmic volume wobble effect.

    Args:
        audio:       Mono float32 audio.
        sample_rate: Hz.
        rate:        LFO rate in Hz (default: 5 = 5 wobbles/second).
    """
    t = np.arange(len(audio)) / sample_rate
    lfo = 0.5 * (1.0 + np.sin(2 * np.pi * rate * t))
    return normalize(audio * lfo).astype(np.float32)
```

### Step 2: Register it

```python
# Add to EFFECTS_REGISTRY dict in effects.py:
"tremolo": {
    "fn": effect_tremolo,
    "label": "🌊 Tremolo",
    "description": "LFO amplitude modulation — rhythmic wobble!",
    "params": {"rate": 5.0},
    "color": "#22c55e",
    "emoji": "🌊",
},
```

### Step 3: Open a Pull Request 🎉

That's it! The UI auto-discovers your effect. No other files need changing.

---

## 🛠️ Tech Stack

| Library | Purpose | License |
|---|---|---|
| [Streamlit](https://streamlit.io) | Web UI framework | Apache 2.0 |
| [NumPy](https://numpy.org) | Audio array math | BSD |
| [SciPy](https://scipy.org) | DSP filters & resampling | BSD |
| [PyDub](https://github.com/jiaaro/pydub) | MP3/OGG audio loading | MIT |

All dependencies are **100% free and open-source**.

---

## 📡 Live Call — Contributor Implementation Guide

The `live_call.py` module contains:
- ✅ Room creation and management
- ✅ Participant tracking
- ✅ Stream chunk processing concept
- 🚧 WebSocket server (needs implementation)

**Suggested approaches:**

```bash
# Option A: websockets library (simplest)
pip install websockets

# Option B: aiortc for WebRTC (full P2P)
pip install aiortc

# Option C: FastAPI WebSocket (integrated)
pip install fastapi uvicorn
```

See the detailed guide inside `live_call.py`.

---

## 🤝 Contributing

1. Fork the repository
2. Create a branch: `git checkout -b feat/my-cool-effect`
3. Make your changes
4. Run the app to test: `streamlit run main.py`
5. Open a Pull Request with:
   - Description of your effect / feature
   - DSP math explanation (for effects)
   - Before/after audio examples (optional but appreciated!)

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

Built for **Hacktoberfest 2026** with ❤️ by the Yash.

# 🤝 Contributing to VoiceMorph

Thank you for contributing to VoiceMorph! This guide will help you get started.

## Code of Conduct

Be kind, welcoming, and respectful. Hacktoberfest is for everyone.

## Types of Contributions

### 🎛️ Adding a Voice Effect (Recommended for beginners!)

See the [Adding an Effect guide in README.md](README.md#-adding-a-custom-voice-effect-hacktoberfest).

TL;DR: Write a function in `effects.py`, add it to `EFFECTS_REGISTRY`, and it automatically appears in the UI.

### 📡 Implementing WebSocket Live Calls

See `live_call.py` — the room management is done, the server just needs wiring up.

### 🐛 Bug Fixes

Open an issue first describing the bug, then submit a PR with:
- What was broken
- How you fixed it
- Test steps

### 🎨 UI Improvements

The UI styles are all in the `<style>` block at the top of `main.py`.

## Pull Request Guidelines

- Keep PRs focused — one feature / fix per PR
- For new effects, include a comment explaining the DSP math
- Test your changes by running `streamlit run main.py`
- Use descriptive commit messages: `feat: add tremolo effect`, `fix: normalize clipping in monster effect`

## DSP Math Resources

- [The Scientist and Engineer's Guide to DSP](https://www.dspguide.com/) — free online book
- [Julius O. Smith's Introduction to Digital Filters](https://ccrma.stanford.edu/~jos/filters/) — Stanford
- [Real-Time Digital Signal Processing](https://numpy.org/doc/stable/) — NumPy docs

## Effect Ideas (unclaimed!)

| Effect | Difficulty | DSP Concept |
|---|---|---|
| Tremolo | ⭐ Easy | LFO amplitude modulation |
| Telephone | ⭐ Easy | Bandpass filter (300–3400 Hz) |
| Underwater | ⭐⭐ Medium | Lowpass + chorus |
| Vibrato | ⭐⭐ Medium | LFO frequency modulation |
| Vocoder | ⭐⭐⭐ Hard | FFT envelope transfer |
| Phase Vocoder | ⭐⭐⭐ Hard | True pitch shifting |
| Chorus | ⭐⭐ Medium | Multi-tap delay + detune |
| Flanger | ⭐⭐ Medium | Short variable delay + feedback |

Happy hacking! 🎃

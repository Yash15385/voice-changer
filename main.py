"""
main.py — Funny Voice Changer & Group Audio Sharing Messenger
==============================================================
🎙️ Hacktoberfest Open Source Project
--------------------------------------
Main Streamlit application routing all pages and components.

Run:
    streamlit run main.py

Pages:
  🎙️ Studio      — Record / upload / synthesize audio + apply 12 DSP effects
  👥 Groups       — Browse friend groups and their shared feeds
  📡 Live Call    — Real-time modulated voice call rooms & WebSocket relay
  ℹ️ About        — Contributor guide and project info
"""

import io
import time
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# Local modules
from effects import EFFECTS_REGISTRY, apply_effect
from audio_utils import (
    SAMPLE_PRESETS,
    TARGET_SAMPLE_RATE,
    TTS_VOICES,
    audio_to_wav_bytes,
    compute_waveform_envelope,
    generate_sample_voice,
    get_audio_info,
    load_audio,
    resample_audio,
    synthesize_text_to_speech,
    trim_silence,
)
from storage import (
    cleanup_old_clips,
    cleanup_temp,
    create_group,
    delete_post,
    get_group,
    get_group_feed,
    get_groups,
    get_storage_stats,
    init_storage,
    post_to_group,
    save_clip,
)
from live_call import (
    WebSocketStub,
    broadcast_voice_to_room,
    create_room,
    end_room,
    get_active_rooms,
    get_room,
    list_room_transmissions,
)

# ---------------------------------------------------------------------------
# PAGE CONFIG & GLOBAL STYLES
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="🎙️ VoiceMorph — Funny Voice Changer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS
st.markdown("""
<style>
/* ── Google Font ─────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;700&display=swap');

/* ── Root palette ────────────────────────────── */
:root {
  --bg-main: #0d0f1a;
  --bg-card: #151826;
  --bg-card-hover: #1c2035;
  --bg-glass: rgba(255,255,255,0.04);
  --border: rgba(255,255,255,0.08);
  --accent: #7c3aed;
  --accent2: #06b6d4;
  --success: #22c55e;
  --warning: #f59e0b;
  --danger: #ef4444;
  --text-primary: #f1f5f9;
  --text-secondary: #94a3b8;
  --text-muted: #475569;
  --radius: 16px;
  --radius-sm: 10px;
}

/* ── Global reset ────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
  background-color: var(--bg-main) !important;
  font-family: 'Inter', sans-serif;
  color: var(--text-primary);
}

[data-testid="stSidebar"] {
  background: #0a0c16 !important;
  border-right: 1px solid var(--border);
}

[data-testid="stHeader"] { background: transparent !important; }

/* ── Streamlit element overrides ─────────────── */
.stButton > button {
  background: linear-gradient(135deg, var(--accent), #9333ea) !important;
  color: white !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  padding: 0.55rem 1.4rem !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 4px 20px rgba(124,58,237,0.35) !important;
}
.stButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 28px rgba(124,58,237,0.5) !important;
}

.stSelectbox > div > div,
.stTextInput > div > div > input,
.stTextArea textarea {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-primary) !important;
  border-radius: var(--radius-sm) !important;
  font-family: 'Inter', sans-serif !important;
}

[data-testid="stFileUploader"] {
  background: var(--bg-card) !important;
  border: 2px dashed var(--border) !important;
  border-radius: var(--radius) !important;
}

.stAudio { border-radius: var(--radius-sm) !important; }

/* ── Custom components ───────────────────────── */
.vm-hero {
  background: linear-gradient(135deg, #1a0533 0%, #0f1729 50%, #001a2e 100%);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 2.2rem 2rem;
  margin-bottom: 1.5rem;
  position: relative;
  overflow: hidden;
}
.vm-hero::before {
  content: '';
  position: absolute;
  top: -60px; right: -60px;
  width: 220px; height: 220px;
  background: radial-gradient(circle, rgba(124,58,237,0.25) 0%, transparent 70%);
  pointer-events: none;
}
.vm-hero h1 {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 2.1rem; font-weight: 700;
  background: linear-gradient(135deg, #c084fc, #67e8f9);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin: 0 0 0.4rem 0;
}
.vm-hero p { color: var(--text-secondary); margin: 0; font-size: 0.98rem; }

.vm-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.3rem;
  margin-bottom: 1rem;
  transition: all 0.2s ease;
}
.vm-card:hover {
  border-color: rgba(124,58,237,0.4);
  background: var(--bg-card-hover);
}

.effect-card {
  background: var(--bg-card);
  border: 2px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem;
  cursor: pointer;
  transition: all 0.2s ease;
  text-align: center;
  min-height: 132px;
}
.effect-card.selected {
  border-color: var(--accent) !important;
  background: rgba(124,58,237,0.12) !important;
  box-shadow: 0 0 0 3px rgba(124,58,237,0.2);
}
.effect-card:hover {
  transform: translateY(-2px);
  border-color: rgba(124,58,237,0.5);
}
.effect-emoji { font-size: 2rem; display: block; margin-bottom: 0.3rem; }
.effect-name { font-weight: 700; font-size: 0.92rem; color: var(--text-primary); }
.effect-desc { font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.25rem; line-height: 1.35; }

.group-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.1rem 1.3rem;
  display: flex; align-items: center; gap: 1rem;
  cursor: pointer;
  transition: all 0.2s ease;
  margin-bottom: 0.65rem;
}
.group-card:hover {
  border-color: rgba(124,58,237,0.4);
  transform: translateX(3px);
}
.group-avatar {
  width: 48px; height: 48px;
  border-radius: 14px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.5rem;
  flex-shrink: 0;
}
.group-info h3 { margin: 0; font-size: 0.98rem; font-weight: 700; }
.group-info p { margin: 0; font-size: 0.8rem; color: var(--text-secondary); }

.feed-item {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.2rem;
  margin-bottom: 0.6rem;
  position: relative;
}
.feed-header {
  display: flex; align-items: center; gap: 0.7rem;
}
.feed-avatar {
  width: 36px; height: 36px;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 0.88rem; color: white;
  flex-shrink: 0;
}
.feed-meta { flex: 1; }
.feed-meta strong { font-size: 0.9rem; color: var(--text-primary); }
.feed-meta small { display: block; color: var(--text-muted); font-size: 0.75rem; }
.feed-effect-tag {
  font-size: 0.75rem; font-weight: 600;
  padding: 0.2rem 0.65rem;
  border-radius: 999px;
  border: 1px solid;
}

.stat-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.2rem;
  text-align: center;
}

.live-badge {
  display: inline-flex; align-items: center; gap: 0.4rem;
  background: rgba(239,68,68,0.15); color: #f87171;
  border: 1px solid rgba(239,68,68,0.3);
  border-radius: 999px; padding: 0.3rem 0.9rem;
  font-size: 0.82rem; font-weight: 700;
  animation: pulse-live 1.5s infinite;
}
.live-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: #ef4444;
  animation: blink 1s infinite;
}
@keyframes blink {
  0%, 100% { opacity: 1; } 50% { opacity: 0.2; }
}
@keyframes pulse-live {
  0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.3); }
  50% { box-shadow: 0 0 0 5px rgba(239,68,68,0); }
}

.section-header {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.12rem; font-weight: 700;
  color: var(--text-primary);
  margin: 1.3rem 0 0.75rem 0;
  display: flex; align-items: center; gap: 0.5rem;
}

.divider {
  border: none;
  border-top: 1px solid var(--border);
  margin: 1.3rem 0;
}

.info-box {
  background: rgba(6,182,212,0.08);
  border: 1px solid rgba(6,182,212,0.2);
  border-radius: var(--radius-sm);
  padding: 0.85rem 1.1rem;
  font-size: 0.86rem;
  color: #67e8f9;
  margin-bottom: 1rem;
}

.success-box {
  background: rgba(34,197,94,0.08);
  border: 1px solid rgba(34,197,94,0.2);
  border-radius: var(--radius-sm);
  padding: 0.85rem 1.1rem;
  font-size: 0.86rem;
  color: #86efac;
  margin-bottom: 1rem;
}

.warn-box {
  background: rgba(245,158,11,0.08);
  border: 1px solid rgba(245,158,11,0.2);
  border-radius: var(--radius-sm);
  padding: 0.85rem 1.1rem;
  font-size: 0.86rem;
  color: #fcd34d;
  margin-bottom: 1rem;
}

/* Contributor banner */
.contrib-banner {
  background: linear-gradient(135deg, rgba(249,115,22,0.15), rgba(239,68,68,0.1));
  border: 1px solid rgba(249,115,22,0.3);
  border-radius: var(--radius);
  padding: 1rem 1.2rem;
  display: flex; align-items: center; gap: 0.9rem;
}
.contrib-banner .icon { font-size: 1.8rem; }
.contrib-banner h4 { margin: 0; font-size: 0.92rem; color: #fb923c; }
.contrib-banner p { margin: 0; font-size: 0.78rem; color: var(--text-secondary); }

/* Waveform decorative bars */
.waveform { display: flex; align-items: center; gap: 3px; height: 32px; }
.waveform .bar {
  width: 4px; border-radius: 2px;
  background: linear-gradient(to top, var(--accent), var(--accent2));
  animation: wave 1.2s ease-in-out infinite;
}
.waveform .bar:nth-child(2) { animation-delay: 0.1s; }
.waveform .bar:nth-child(3) { animation-delay: 0.2s; }
.waveform .bar:nth-child(4) { animation-delay: 0.3s; }
.waveform .bar:nth-child(5) { animation-delay: 0.2s; }
.waveform .bar:nth-child(6) { animation-delay: 0.1s; }
@keyframes wave {
  0%, 100% { height: 8px; } 50% { height: 28px; }
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# INIT SESSION STATE
# ---------------------------------------------------------------------------

init_storage()

if "page" not in st.session_state:
    st.session_state.page = "studio"
if "username" not in st.session_state:
    st.session_state.username = "Anonymous"
if "selected_effect" not in st.session_state:
    st.session_state.selected_effect = "helium"
if "effect_params" not in st.session_state:
    st.session_state.effect_params = {}
if "sample_audio_bytes" not in st.session_state:
    st.session_state.sample_audio_bytes = None
if "sample_audio_name" not in st.session_state:
    st.session_state.sample_audio_name = None
if "processed_audio" not in st.session_state:
    st.session_state.processed_audio = None
if "processed_meta" not in st.session_state:
    st.session_state.processed_meta = None
if "active_group" not in st.session_state:
    groups_init = get_groups()
    st.session_state.active_group = groups_init[0]["id"] if groups_init else None
if "active_room" not in st.session_state:
    st.session_state.active_room = None


def _render_waveform_bars(envelope: list[float], color_hex: str, label: str) -> None:
    """Render an HTML/CSS bar waveform visualization."""
    bars_html = "".join(
        f'<div style="flex:1; height:{max(6, int(v * 54))}px; '
        f'background:{color_hex}; border-radius:2px; opacity:0.88;"></div>'
        for v in envelope
    )
    st.markdown(
        f"""
        <div style="background:#0a0c16; border:1px solid rgba(255,255,255,0.07);
                    border-radius:10px; padding:0.65rem 0.85rem; margin-bottom:0.65rem;">
          <div style="font-size:0.72rem; color:#94a3b8; font-weight:600; margin-bottom:0.35rem;">
            {label}
          </div>
          <div style="display:flex; align-items:center; gap:2px; height:58px;">
            {bars_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style="padding: 0.8rem 0 1.2rem 0; text-align: center;">
      <div style="font-size: 2.4rem; margin-bottom: 0.25rem;">🎙️</div>
      <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.3rem;
                  font-weight: 700; background: linear-gradient(135deg, #c084fc, #67e8f9);
                  -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        VoiceMorph
      </div>
      <div style="font-size: 0.75rem; color: #475569; margin-top: 0.15rem;">
        Funny Voice Changer &amp; Messenger
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">👤 Your Name</div>', unsafe_allow_html=True)
    username_input = st.text_input(
        "Display name",
        value=st.session_state.username,
        label_visibility="collapsed",
        placeholder="Enter your display name…",
        key="username_field",
    )
    if username_input:
        st.session_state.username = username_input

    st.markdown('<hr class="divider" style="margin: 0.9rem 0;">', unsafe_allow_html=True)

    st.markdown('<div class="section-header">🧭 Navigation</div>', unsafe_allow_html=True)

    pages = [
        ("studio", "🎙️", "Studio"),
        ("groups", "👥", "Groups"),
        ("live", "📡", "Live Call"),
        ("about", "ℹ️", "About"),
    ]

    for page_key, icon, label in pages:
        if st.button(
            f"{icon} {label}",
            key=f"nav_{page_key}",
            use_container_width=True,
        ):
            st.session_state.page = page_key
            st.rerun()

    st.markdown('<hr class="divider" style="margin: 0.9rem 0;">', unsafe_allow_html=True)

    stats = get_storage_stats()
    st.markdown(f"""
    <div class="vm-card" style="padding: 0.95rem;">
      <div style="font-size: 0.76rem; color: #475569; font-weight: 700; margin-bottom: 0.55rem;">
        💾 STORAGE STATS
      </div>
      <div style="display: flex; justify-content: space-between; margin-bottom: 0.28rem;">
        <span style="font-size: 0.82rem; color: #94a3b8;">Clips stored</span>
        <span style="font-size: 0.82rem; font-weight: 600; color: #c084fc;">{stats['clip_count']}</span>
      </div>
      <div style="display: flex; justify-content: space-between; margin-bottom: 0.28rem;">
        <span style="font-size: 0.82rem; color: #94a3b8;">Total size</span>
        <span style="font-size: 0.82rem; font-weight: 600; color: #67e8f9;">{stats['total_mb']} MB</span>
      </div>
      <div style="display: flex; justify-content: space-between;">
        <span style="font-size: 0.82rem; color: #94a3b8;">Groups</span>
        <span style="font-size: 0.82rem; font-weight: 600; color: #86efac;">{stats['groups']}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🧹 Cleanup Old Clips", use_container_width=True, key="cleanup_btn"):
        n = cleanup_old_clips()
        cleanup_temp()
        st.success(f"Deleted {n} old clip(s).")

    st.markdown("""
    <div class="contrib-banner" style="margin-top: 0.8rem;">
      <div class="icon">🎃</div>
      <div>
        <h4>12 DSP Effects Ready</h4>
        <p>Studio, Group Feeds &amp; Live Call Rooms active!</p>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# ─────────────────────────── PAGE: STUDIO ──────────────────────────────────
# ---------------------------------------------------------------------------

def page_studio():
    st.markdown("""
    <div class="vm-hero">
      <div style="display:flex; align-items:center; gap: 1rem;">
        <div>
          <h1>🎙️ Voice Studio</h1>
          <p>Upload, record, or synthesize a voice clip, customize 12 DSP effects, and share with your squad.</p>
        </div>
        <div class="waveform" style="margin-left: auto;">
          <div class="bar" style="height:10px"></div>
          <div class="bar" style="height:22px"></div>
          <div class="bar" style="height:16px"></div>
          <div class="bar" style="height:28px"></div>
          <div class="bar" style="height:18px"></div>
          <div class="bar" style="height:12px"></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1.15, 0.85], gap="large")

    # ── LEFT: Input & Effect Controls ────────────────────────────────────────
    with col_left:
        st.markdown('<div class="section-header">📁 Audio Input</div>', unsafe_allow_html=True)

        input_tab_tts, input_tab1, input_tab2, input_tab3 = st.tabs([
            "✍️ Custom Text to Speech",
            "📁 Upload File",
            "🎤 Record (Browser)",
            "🎹 Sample Phrases",
        ])

        uploaded_file = None
        recorded_audio = None

        with input_tab_tts:
            st.markdown("""
            <div class="info-box">
              ✍️ <strong>Type Any Custom Text:</strong> Enter your own words below, pick a voice persona, and apply any of the 12 clear voice effects!
            </div>
            """, unsafe_allow_html=True)
            custom_text_val = st.text_area(
                "Custom text to speak",
                value="Hello squad! This is my custom text in VoiceMorph, and every word is super clear!",
                height=85,
                placeholder="Type any sentence you want to morph...",
                key="custom_tts_textarea",
            )
            tts_col1, tts_col2 = st.columns([1.4, 1])
            with tts_col1:
                chosen_tts_voice = st.selectbox(
                    "Voice Persona",
                    options=list(TTS_VOICES.keys()),
                    format_func=lambda k: TTS_VOICES[k],
                    key="tts_voice_select",
                )
            with tts_col2:
                chosen_tts_speed = st.slider(
                    "Speech Speed",
                    min_value=0.75,
                    max_value=1.35,
                    value=1.0,
                    step=0.05,
                    key="tts_speed_slider",
                )
            if st.button("🗣️ Generate Speech from Custom Text", use_container_width=True, key="gen_custom_tts_btn"):
                arr, sr = synthesize_text_to_speech(custom_text_val, voice=chosen_tts_voice, speed=chosen_tts_speed)
                st.session_state.sample_audio_bytes = audio_to_wav_bytes(arr, sr)
                st.session_state.sample_audio_name = f"custom_tts_{chosen_tts_voice}.wav"
                st.rerun()
            if st.session_state.sample_audio_bytes:
                st.audio(io.BytesIO(st.session_state.sample_audio_bytes), format="audio/wav")

        with input_tab1:
            uploaded_file = st.file_uploader(
                "Drop your audio here",
                type=["wav", "mp3", "ogg", "flac"],
                label_visibility="collapsed",
                key="audio_upload",
            )
            if uploaded_file:
                st.audio(uploaded_file, format="audio/wav")
                st.markdown(f"""
                <div class="vm-card" style="padding: 0.75rem 1rem; margin-top: 0.4rem;">
                  <span style="font-size: 0.85rem; color: #94a3b8;">📄 {uploaded_file.name}</span>
                  <span style="float:right; font-size: 0.85rem; color: #67e8f9; font-weight: 600;">
                    {round(uploaded_file.size / 1024, 1)} KB
                  </span>
                </div>
                """, unsafe_allow_html=True)

        with input_tab2:
            st.markdown("""
            <div class="info-box">
              🎤 <strong>Browser Recording:</strong> Click the microphone button below to record your voice directly in the browser.
            </div>
            """, unsafe_allow_html=True)
            recorded_audio = st.audio_input(
                "🎤 Click to record",
                key="mic_recorder",
                label_visibility="collapsed",
            )
            if recorded_audio:
                st.audio(recorded_audio, format="audio/wav")

        with input_tab3:
            st.markdown("""
            <div class="info-box">
              🎹 <strong>No mic or file handy?</strong> Generate a synthesized vocal phrase below to test all 12 voice effects immediately!
            </div>
            """, unsafe_allow_html=True)
            preset_options = {k: v["label"] for k, v in SAMPLE_PRESETS.items()}
            chosen_preset = st.selectbox(
                "Choose a sample vocal phrase",
                options=list(preset_options.keys()),
                format_func=lambda k: preset_options[k],
                label_visibility="collapsed",
                key="sample_preset_select",
            )
            if st.button("🎵 Load Sample Voice Clip", use_container_width=True, key="gen_sample_btn"):
                arr, sr = generate_sample_voice(chosen_preset, TARGET_SAMPLE_RATE)
                st.session_state.sample_audio_bytes = audio_to_wav_bytes(arr, sr)
                st.session_state.sample_audio_name = f"sample_{chosen_preset}.wav"
                st.rerun()

            if st.session_state.sample_audio_bytes:
                st.audio(io.BytesIO(st.session_state.sample_audio_bytes), format="audio/wav")
                st.caption(f"Loaded sample clip: `{st.session_state.sample_audio_name}`")

        # ── Effect Picker ──────────────────────────────────────────────────
        st.markdown('<div class="section-header">🎛️ Choose Your Effect (12 DSP Filters)</div>', unsafe_allow_html=True)

        effect_keys = list(EFFECTS_REGISTRY.keys())
        effect_cols = st.columns(3)

        for idx, key in enumerate(effect_keys):
            meta = EFFECTS_REGISTRY[key]
            col = effect_cols[idx % 3]
            is_selected = st.session_state.selected_effect == key
            selected_cls = "selected" if is_selected else ""
            rgb_str = ",".join([str(int(meta["color"][i : i + 2], 16)) for i in (1, 3, 5)])

            with col:
                st.markdown(f"""
                <div class="effect-card {selected_cls}"
                     style="border-color: {meta['color'] if is_selected else 'rgba(255,255,255,0.08)'};
                            background: {'rgba(' + rgb_str + ',0.14)' if is_selected else 'var(--bg-card)'};">
                  <span class="effect-emoji">{meta['emoji']}</span>
                  <div class="effect-name">{meta['label']}</div>
                  <div class="effect-desc">{meta['description']}</div>
                </div>
                """, unsafe_allow_html=True)

                btn_label = "✓ Selected" if is_selected else "Select"
                if st.button(btn_label, key=f"eff_{key}", use_container_width=True):
                    st.session_state.selected_effect = key
                    st.rerun()

        # ── Fine-Tune Effect Parameters ────────────────────────────────────
        selected_key = st.session_state.selected_effect
        selected_meta = EFFECTS_REGISTRY[selected_key]
        param_schema = selected_meta.get("param_schema", {})
        current_overrides = {}

        if param_schema:
            with st.expander(f"🎚️ Fine-Tune {selected_meta['label']} Parameters", expanded=True):
                p_cols = st.columns(min(3, len(param_schema)))
                for idx, (p_name, p_cfg) in enumerate(param_schema.items()):
                    with p_cols[idx % len(p_cols)]:
                        default_val = p_cfg["default"]
                        is_int = isinstance(default_val, int) and isinstance(p_cfg["step"], int)
                        val = st.slider(
                            p_cfg["label"],
                            min_value=int(p_cfg["min"]) if is_int else float(p_cfg["min"]),
                            max_value=int(p_cfg["max"]) if is_int else float(p_cfg["max"]),
                            value=int(default_val) if is_int else float(default_val),
                            step=int(p_cfg["step"]) if is_int else float(p_cfg["step"]),
                            key=f"param_{selected_key}_{p_name}",
                        )
                        current_overrides[p_name] = val

        # ── Process Button ─────────────────────────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        has_source = (
            uploaded_file is not None
            or recorded_audio is not None
            or st.session_state.sample_audio_bytes is not None
        )
        if not has_source:
            st.markdown("""
            <div class="warn-box">
              ⬆️ Upload a file, record from your mic, or load a Sample Voice above to enable processing.
            </div>
            """, unsafe_allow_html=True)

        if st.button(
            f"✨ Apply {selected_meta['label']} Effect",
            key="process_btn",
            use_container_width=True,
            disabled=not has_source,
        ):
            with st.spinner(f"Applying {selected_meta['label']}…"):
                try:
                    if uploaded_file is not None:
                        raw_bytes = uploaded_file.getvalue()
                        filename = uploaded_file.name
                    elif recorded_audio is not None:
                        raw_bytes = recorded_audio.getvalue()
                        filename = getattr(recorded_audio, "name", "mic_recording.wav")
                    else:
                        raw_bytes = st.session_state.sample_audio_bytes
                        filename = st.session_state.sample_audio_name or "sample_voice.wav"

                    audio, sr = load_audio(raw_bytes, filename)
                    if sr != TARGET_SAMPLE_RATE:
                        audio = resample_audio(audio, sr, TARGET_SAMPLE_RATE)
                        sr = TARGET_SAMPLE_RATE

                    audio = trim_silence(audio, sample_rate=sr)
                    orig_env = compute_waveform_envelope(audio, num_points=96)

                    processed = apply_effect(
                        selected_key,
                        audio,
                        sr,
                        **current_overrides,
                    )
                    proc_env = compute_waveform_envelope(processed, num_points=96)
                    wav_bytes = audio_to_wav_bytes(processed, sr)
                    info = get_audio_info(processed, sr)

                    st.session_state.processed_audio = wav_bytes
                    st.session_state.processed_meta = {
                        "effect": selected_key,
                        "filename": filename,
                        "info": info,
                        "orig_envelope": orig_env,
                        "proc_envelope": proc_env,
                        "params": current_overrides,
                    }
                    st.success("🎉 Effect applied!")
                    time.sleep(0.15)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Processing failed: {e}")

    # ── RIGHT: Output ────────────────────────────────────────────────────────
    with col_right:
        st.markdown('<div class="section-header">🔊 Processed Output</div>', unsafe_allow_html=True)

        if st.session_state.processed_audio and st.session_state.processed_meta:
            meta = st.session_state.processed_meta
            info = meta["info"]
            eff_meta = EFFECTS_REGISTRY[meta["effect"]]

            st.markdown(f"""
            <div class="vm-card">
              <div style="display: flex; align-items: center; gap: 0.8rem; margin-bottom: 1rem;">
                <span style="font-size: 2rem;">{eff_meta['emoji']}</span>
                <div>
                  <div style="font-weight: 700; font-size: 1rem;">{eff_meta['label']} Applied</div>
                  <div style="font-size: 0.8rem; color: #94a3b8;">{meta['filename']}</div>
                </div>
                <div style="margin-left: auto; background: rgba(124,58,237,0.2);
                            color: #c084fc; border-radius: 999px; padding: 0.2rem 0.8rem;
                            font-size: 0.78rem; font-weight: 700;">
                  PROCESSED
                </div>
              </div>
              <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                <div style="background: var(--bg-main); border-radius: 8px; padding: 0.6rem 0.8rem;">
                  <div style="font-size: 0.72rem; color: #475569;">DURATION</div>
                  <div style="font-weight: 700; font-size: 0.95rem; color: #c084fc;">
                    {info['duration_sec']}s
                  </div>
                </div>
                <div style="background: var(--bg-main); border-radius: 8px; padding: 0.6rem 0.8rem;">
                  <div style="font-size: 0.72rem; color: #475569;">SAMPLE RATE</div>
                  <div style="font-weight: 700; font-size: 0.95rem; color: #67e8f9;">
                    {info['sample_rate']} Hz
                  </div>
                </div>
                <div style="background: var(--bg-main); border-radius: 8px; padding: 0.6rem 0.8rem;">
                  <div style="font-size: 0.72rem; color: #475569;">PEAK</div>
                  <div style="font-weight: 700; font-size: 0.95rem; color: #86efac;">
                    {info['peak_amplitude']}
                  </div>
                </div>
                <div style="background: var(--bg-main); border-radius: 8px; padding: 0.6rem 0.8rem;">
                  <div style="font-size: 0.72rem; color: #475569;">RMS</div>
                  <div style="font-weight: 700; font-size: 0.95rem; color: #fbbf24;">
                    {info['rms']}
                  </div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            if meta.get("orig_envelope") and meta.get("proc_envelope"):
                _render_waveform_bars(meta["orig_envelope"], "#64748b", "INPUT WAVEFORM ENVELOPE")
                _render_waveform_bars(meta["proc_envelope"], eff_meta["color"], f"MORPHEUS DSP OUTPUT ({eff_meta['label'].upper()})")

            st.audio(io.BytesIO(st.session_state.processed_audio), format="audio/wav")

            st.download_button(
                label="⬇️ Download Processed Clip (.wav)",
                data=st.session_state.processed_audio,
                file_name=f"voicemorph_{meta['effect']}_{int(time.time())}.wav",
                mime="audio/wav",
                use_container_width=True,
            )

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">📤 Share to Group</div>', unsafe_allow_html=True)

            groups = get_groups()
            group_options = {g["name"]: g["id"] for g in groups}

            selected_group_name = st.selectbox(
                "Choose a group",
                list(group_options.keys()),
                label_visibility="collapsed",
                key="share_group_select",
            )

            if st.button("🚀 Share Clip to Group", use_container_width=True, key="share_btn"):
                group_id = group_options[selected_group_name]
                clip_meta = save_clip(
                    audio_bytes=st.session_state.processed_audio,
                    effect=meta["effect"],
                    author=st.session_state.username,
                    original_filename=meta["filename"],
                )
                post_to_group(group_id, clip_meta)
                st.session_state.active_group = group_id
                st.markdown(f"""
                <div class="success-box">
                  🎉 Shared to <strong>{selected_group_name}</strong>!
                  Switch to <strong>Groups</strong> to listen in the squad feed.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="vm-card" style="text-align: center; padding: 3rem 2rem;">
              <div style="font-size: 3.5rem; margin-bottom: 1rem;">🎵</div>
              <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 0.5rem;">
                No clip processed yet
              </div>
              <div style="font-size: 0.88rem; color: #94a3b8;">
                Upload audio, record from your mic, or use the <strong>Sample Voice</strong> tab,<br>
                pick any of the 12 effects, and click <strong>"Apply Effect"</strong>.
              </div>
            </div>
            """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# ─────────────────────────── PAGE: GROUPS ──────────────────────────────────
# ---------------------------------------------------------------------------

def page_groups():
    st.markdown("""
    <div class="vm-hero">
      <h1>👥 Friend Groups</h1>
      <p>Share modulated voice clips with your squads and listen to the live group feed.</p>
    </div>
    """, unsafe_allow_html=True)

    col_groups, col_feed = st.columns([1, 1.6], gap="large")

    with col_groups:
        st.markdown('<div class="section-header">🏠 Your Groups</div>', unsafe_allow_html=True)

        groups = get_groups()
        for group in groups:
            feed = get_group_feed(group["id"])
            is_active = st.session_state.active_group == group["id"]
            border_style = "border-color: #7c3aed;" if is_active else ""

            st.markdown(f"""
            <div class="group-card" style="{border_style}">
              <div class="group-avatar" style="background: {group['color']}22;">
                {group['emoji']}
              </div>
              <div class="group-info">
                <h3>{group['name']}</h3>
                <p>{group['description']}</p>
                <p style="margin-top: 0.3rem; font-size: 0.75rem; color: #475569;">
                  👥 {group['member_count']} members • 🎵 {len(feed)} clips
                </p>
              </div>
              {'<div style="width:8px;height:8px;border-radius:50%;background:#7c3aed;flex-shrink:0;"></div>' if is_active else ''}
            </div>
            """, unsafe_allow_html=True)

            if st.button(
                "📂 Open Feed" if not is_active else "✓ Viewing",
                key=f"open_group_{group['id']}",
                use_container_width=True,
            ):
                st.session_state.active_group = group["id"]
                st.rerun()

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">➕ Create Group</div>', unsafe_allow_html=True)

        with st.form("create_group_form", clear_on_submit=True):
            new_name = st.text_input("Group name", placeholder="e.g. The Night Owls 🦉")
            new_desc = st.text_input("Description", placeholder="What's this group about?")
            new_emoji = st.text_input("Emoji", value="🔊", max_chars=4)
            submitted = st.form_submit_button("Create Group", use_container_width=True)
            if submitted and new_name.strip():
                created = create_group(new_name.strip(), new_desc.strip(), new_emoji)
                st.session_state.active_group = created["id"]
                st.success(f"✅ Group '{new_name}' created!")
                st.rerun()

    with col_feed:
        if st.session_state.active_group:
            group = get_group(st.session_state.active_group)
            feed = get_group_feed(st.session_state.active_group)

            if group:
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap: 0.8rem; margin-bottom: 1.2rem;">
                  <div style="font-size: 2rem;">{group['emoji']}</div>
                  <div>
                    <div style="font-weight: 700; font-size: 1.1rem;">{group['name']}</div>
                    <div style="font-size: 0.82rem; color: #94a3b8;">{len(feed)} shared clips</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                if not feed:
                    st.markdown("""
                    <div class="vm-card" style="text-align:center; padding: 2.5rem 1.5rem;">
                      <div style="font-size: 3rem; margin-bottom: 0.8rem;">📭</div>
                      <div style="font-weight: 600; margin-bottom: 0.5rem;">No clips yet!</div>
                      <div style="font-size: 0.85rem; color: #94a3b8;">
                        Go to Studio, process a voice clip, and share it here.
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    for clip in feed:
                        effect_key = clip.get("effect", "helium")
                        eff_meta = EFFECTS_REGISTRY.get(effect_key, EFFECTS_REGISTRY["helium"])
                        author = clip.get("author", "Anonymous")
                        initials = author[:2].upper()
                        created_at = clip.get("created_at", "")[:16].replace("T", " ")
                        size_kb = round(clip.get("size_bytes", 0) / 1024, 1)

                        st.markdown(f"""
                        <div class="feed-item">
                          <div class="feed-header">
                            <div class="feed-avatar">{initials}</div>
                            <div class="feed-meta">
                              <strong>{author}</strong>
                              <small>{created_at} UTC · {size_kb} KB</small>
                            </div>
                            <div class="feed-effect-tag" style="color: {eff_meta['color']}; border-color: {eff_meta['color']}55;">
                              {eff_meta['emoji']} {eff_meta['label'].split(' ', 1)[-1]}
                            </div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                        clip_path = clip.get("path")
                        if clip_path and Path(clip_path).exists():
                            audio_bytes = Path(clip_path).read_bytes()
                            st.audio(io.BytesIO(audio_bytes), format="audio/wav")

                            col_dl, col_del = st.columns([3, 1])
                            with col_dl:
                                st.download_button(
                                    "⬇️ Download",
                                    data=audio_bytes,
                                    file_name=f"voicemorph_{clip['id'][:8]}.wav",
                                    mime="audio/wav",
                                    key=f"dl_{clip['id']}",
                                    use_container_width=True,
                                )
                            with col_del:
                                if st.button("🗑️", key=f"del_{clip['id']}", use_container_width=True):
                                    delete_post(st.session_state.active_group, clip["id"])
                                    st.rerun()
                        else:
                            st.markdown("""
                            <div style="font-size: 0.82rem; color: #ef4444; padding: 0.4rem 0;">
                              ⚠️ Audio file not found (may have been cleaned up)
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown('<hr class="divider" style="margin: 0.6rem 0;">', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# ─────────────────────────── PAGE: LIVE CALL ───────────────────────────────
# ---------------------------------------------------------------------------

def page_live():
    st.markdown("""
    <div class="vm-hero">
      <div style="display:flex; align-items:center; gap: 1rem;">
        <div>
          <h1>📡 Live Voice Call Rooms</h1>
          <p>Real-time voice modulation rooms with live push-to-talk broadcasting &amp; WebAudio mic streaming.</p>
        </div>
        <div class="live-badge" style="margin-left: auto; white-space: nowrap;">
          <div class="live-dot"></div> ENGINE ONLINE
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_create, col_rooms = st.columns([0.95, 1.35], gap="large")

    effect_label_to_key = {EFFECTS_REGISTRY[k]["label"]: k for k in EFFECTS_REGISTRY}

    with col_create:
        st.markdown('<div class="section-header">🎙️ Start a Room</div>', unsafe_allow_html=True)

        groups = get_groups()
        group_options = {g["name"]: g["id"] for g in groups}

        call_group = st.selectbox(
            "Select group",
            list(group_options.keys()),
            label_visibility="collapsed",
            key="call_group_select",
        )
        call_effect_label = st.selectbox(
            "Voice effect for the call",
            list(effect_label_to_key.keys()),
            label_visibility="collapsed",
            key="call_effect_select",
        )

        if st.button("📡 Start Call Room", use_container_width=True, key="start_room_btn"):
            group_id = group_options[call_group]
            effect_key = effect_label_to_key[call_effect_label]
            room = create_room(
                group_id=group_id,
                host=st.session_state.username,
                default_effect=effect_key,
            )
            st.session_state.active_room = room.room_id
            st.rerun()

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🔗 Join a Room</div>', unsafe_allow_html=True)

        join_room_id = st.text_input(
            "Room ID",
            placeholder="Paste a room ID (e.g. room_a1b2c3d4)…",
            label_visibility="collapsed",
            key="join_room_input",
        )
        if st.button("Join Room", use_container_width=True, key="join_room_btn"):
            room = get_room(join_room_id.strip())
            if room:
                room.add_participant(
                    st.session_state.username,
                    effect=effect_label_to_key[call_effect_label],
                )
                st.session_state.active_room = room.room_id
                st.success(f"Joined room {room.room_id}!")
                st.rerun()
            else:
                st.error("Room not found or has ended.")

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🟢 Active Rooms</div>', unsafe_allow_html=True)

        active_rooms = get_active_rooms()
        if not active_rooms:
            st.markdown("""
            <div class="vm-card" style="text-align:center; padding: 1.6rem 1.2rem;">
              <div style="font-size: 2rem; margin-bottom: 0.4rem;">📭</div>
              <div style="font-weight: 600;">No active rooms yet</div>
              <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.25rem;">
                Click <strong>Start Call Room</strong> above to open a live voice room.
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for room_dict in active_rooms:
                grp = get_group(room_dict["group_id"]) or {}
                is_current = room_dict["room_id"] == st.session_state.active_room
                eff_meta = EFFECTS_REGISTRY.get(room_dict.get("default_effect", "helium"), EFFECTS_REGISTRY["helium"])

                st.markdown(f"""
                <div class="vm-card" style="{'border-color: #7c3aed;' if is_current else ''}">
                  <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.5rem;">
                    <div class="live-badge"><div class="live-dot"></div> LIVE</div>
                    <div style="font-weight: 700; font-size: 0.92rem;">
                      {grp.get('emoji', '🔊')} {grp.get('name', room_dict['group_id'])}
                    </div>
                  </div>
                  <div style="font-size: 0.8rem; color: #94a3b8;">
                    ID: <code>{room_dict['room_id']}</code> · 👥 {room_dict['participant_count']} · {eff_meta['emoji']} {eff_meta['label']}
                  </div>
                </div>
                """, unsafe_allow_html=True)

                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    if st.button(
                        "🎧 Enter Room" if not is_current else "✓ Inside Room",
                        key=f"enter_{room_dict['room_id']}",
                        use_container_width=True,
                    ):
                        r_obj = get_room(room_dict["room_id"])
                        if r_obj:
                            r_obj.add_participant(st.session_state.username)
                        st.session_state.active_room = room_dict["room_id"]
                        st.rerun()
                with b_col2:
                    if st.button("📴 End Room", key=f"end_{room_dict['room_id']}", use_container_width=True):
                        end_room(room_dict["room_id"])
                        if st.session_state.active_room == room_dict["room_id"]:
                            st.session_state.active_room = None
                        st.rerun()

    # ── RIGHT: Active Room Live Voice Console ────────────────────────────────
    with col_rooms:
        st.markdown('<div class="section-header">🎧 Live Call Console</div>', unsafe_allow_html=True)

        current_room = get_room(st.session_state.active_room) if st.session_state.active_room else None
        if not current_room:
            st.markdown("""
            <div class="vm-card" style="text-align:center; padding: 3rem 2rem;">
              <div style="font-size: 3.2rem; margin-bottom: 0.8rem;">🎙️</div>
              <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 0.4rem;">
                No Call Room Selected
              </div>
              <div style="font-size: 0.86rem; color: #94a3b8;">
                Start a new room on the left or click <strong>Enter Room</strong> on an active room to begin talking with real-time voice effects.
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            grp = get_group(current_room.group_id) or {}
            conn_info = WebSocketStub.get_connection_info(current_room.room_id)
            my_effect = current_room.get_participant_effect(st.session_state.username)
            my_eff_meta = EFFECTS_REGISTRY.get(my_effect, EFFECTS_REGISTRY["helium"])

            st.markdown(f"""
            <div class="vm-card" style="border-color: rgba(124,58,237,0.45);">
              <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.8rem;">
                <div>
                  <span style="font-size:1.3rem;">{grp.get('emoji', '🔊')}</span>
                  <strong style="font-size:1.05rem; margin-left:0.4rem;">{grp.get('name', 'Squad Room')}</strong>
                  <span style="font-size:0.8rem; color:#94a3b8; margin-left:0.5rem;">({current_room.room_id})</span>
                </div>
                <div class="live-badge"><div class="live-dot"></div> CONNECTED</div>
              </div>
              <div style="font-size:0.82rem; color:#94a3b8; margin-bottom:0.6rem;">
                <strong>WebSocket Stream Endpoint:</strong> <code>{conn_info['ws_url']}</code>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Participant badges
            part_badges = []
            for p in current_room.participants:
                p_eff = EFFECTS_REGISTRY.get(p.get("effect", "helium"), EFFECTS_REGISTRY["helium"])
                part_badges.append(
                    f'<span style="display:inline-flex; align-items:center; gap:0.35rem; '
                    f'background:#0a0c16; border:1px solid {p_eff["color"]}66; border-radius:999px; '
                    f'padding:0.25rem 0.75rem; font-size:0.78rem; margin:0.2rem;">'
                    f'🟢 <strong>{p["name"]}</strong> · {p_eff["emoji"]} {p_eff["label"].split(" ", 1)[-1]}</span>'
                )
            st.markdown(
                f'<div style="margin-bottom:0.9rem;">{"".join(part_badges)}</div>',
                unsafe_allow_html=True,
            )

            # Change my active room voice effect
            effect_keys_list = list(EFFECTS_REGISTRY.keys())
            new_eff_key = st.selectbox(
                "🎛️ Your Active Call Voice Effect",
                options=effect_keys_list,
                index=effect_keys_list.index(my_effect) if my_effect in effect_keys_list else 0,
                format_func=lambda k: f"{EFFECTS_REGISTRY[k]['label']} — {EFFECTS_REGISTRY[k]['description']}",
                key="live_my_effect_select",
            )
            if new_eff_key != my_effect:
                current_room.set_participant_effect(st.session_state.username, new_eff_key)
                st.rerun()

            # Push-to-Talk & Real-Time WebAudio Tabs
            call_tab1, call_tab2 = st.tabs([
                "🎙️ Push-to-Talk Room Broadcast",
                "⚡ Real-Time Browser Mic Monitor (Zero-Latency)",
            ])

            with call_tab1:
                st.markdown("""
                <div class="info-box">
                  🎙️ Record a voice message or transmit a test phrase — it will be modulated with your active effect and broadcast to everyone in the room!
                </div>
                """, unsafe_allow_html=True)
                live_mic = st.audio_input(
                    "Record push-to-talk transmission",
                    key=f"ptt_mic_{current_room.room_id}",
                    label_visibility="collapsed",
                )
                ptt_col1, ptt_col2 = st.columns(2)
                with ptt_col1:
                    if st.button(
                        f"📤 Broadcast Mic ({EFFECTS_REGISTRY[new_eff_key]['emoji']})",
                        use_container_width=True,
                        disabled=live_mic is None,
                        key="send_ptt_mic_btn",
                    ):
                        broadcast_voice_to_room(
                            room_id=current_room.room_id,
                            raw_audio_bytes=live_mic.getvalue(),
                            sender=st.session_state.username,
                            effect_key=new_eff_key,
                            filename="ptt_mic.wav",
                        )
                        st.success("Transmitted to room!")
                        st.rerun()
                with ptt_col2:
                    if st.button(
                        f"🎹 Send Sample Voice ({EFFECTS_REGISTRY[new_eff_key]['emoji']})",
                        use_container_width=True,
                        key="send_ptt_sample_btn",
                    ):
                        arr, sr = generate_sample_voice("greeting", TARGET_SAMPLE_RATE)
                        wav_b = audio_to_wav_bytes(arr, sr)
                        broadcast_voice_to_room(
                            room_id=current_room.room_id,
                            raw_audio_bytes=wav_b,
                            sender=st.session_state.username,
                            effect_key=new_eff_key,
                            filename="sample_transmission.wav",
                        )
                        st.success("Sample voice modulated & broadcast to room!")
                        st.rerun()

            with call_tab2:
                # Embedded WebAudio real-time live modulator widget
                components.html(
                    f"""
                    <div style="background:#0a0c16; color:#f1f5f9; font-family:'Inter',sans-serif;
                                border:1px solid rgba(124,58,237,0.35); border-radius:12px; padding:14px;">
                      <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:10px;">
                        <div style="font-weight:700; font-size:14px;">
                          ⚡ Live WebAudio Microphone Modulator ({EFFECTS_REGISTRY[new_eff_key]['label']})
                        </div>
                        <button id="toggleMicBtn" style="background:linear-gradient(135deg,#7c3aed,#9333ea);
                                color:white; border:none; border-radius:8px; padding:6px 14px;
                                font-weight:600; cursor:pointer; font-size:13px;">
                          🎙️ Start Live Mic
                        </button>
                      </div>
                      <canvas id="liveCanvas" width="480" height="64"
                              style="width:100%; height:64px; background:#151826; border-radius:8px;"></canvas>
                      <div id="liveStatus" style="font-size:12px; color:#94a3b8; margin-top:6px;">
                        Click "Start Live Mic" to hear your modulated voice in real time (use headphones to avoid feedback).
                      </div>
                    </div>
                    <script>
                    (function(){{
                      const btn = document.getElementById('toggleMicBtn');
                      const status = document.getElementById('liveStatus');
                      const canvas = document.getElementById('liveCanvas');
                      const ctx2d = canvas.getContext('2d');
                      let audioCtx = null, stream = null, animId = null, active = false;

                      btn.onclick = async () => {{
                        if (active) {{
                          if (stream) stream.getTracks().forEach(t => t.stop());
                          if (audioCtx) audioCtx.close();
                          cancelAnimationFrame(animId);
                          active = false;
                          btn.textContent = '🎙️ Start Live Mic';
                          status.textContent = 'Live mic stopped.';
                          return;
                        }}
                        try {{
                          stream = await navigator.mediaDevices.getUserMedia({{audio: true}});
                          audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                          const src = audioCtx.createMediaStreamSource(stream);
                          const analyser = audioCtx.createAnalyser();
                          analyser.fftSize = 128;

                          // Modulator node chain
                          const filter = audioCtx.createBiquadFilter();
                          filter.type = 'bandpass';
                          filter.frequency.value = 1100;
                          const delay = audioCtx.createDelay();
                          delay.delayTime.value = 0.08;
                          const osc = audioCtx.createOscillator();
                          const lfoGain = audioCtx.createGain();
                          osc.frequency.value = 6.0;
                          lfoGain.gain.value = 0.004;
                          osc.connect(lfoGain);
                          lfoGain.connect(delay.delayTime);
                          osc.start();

                          src.connect(filter);
                          filter.connect(delay);
                          delay.connect(analyser);
                          analyser.connect(audioCtx.destination);

                          active = true;
                          btn.textContent = '⏹️ Stop Live Mic';
                          status.textContent = '🟢 Streaming live modulated audio to speakers & WebSocket room...';

                          const data = new Uint8Array(analyser.frequencyBinCount);
                          function draw() {{
                            if (!active) return;
                            animId = requestAnimationFrame(draw);
                            analyser.getByteFrequencyData(data);
                            ctx2d.fillStyle = '#151826';
                            ctx2d.fillRect(0, 0, canvas.width, canvas.height);
                            const barW = (canvas.width / data.length) - 2;
                            for (let i = 0; i < data.length; i++) {{
                              const h = (data[i] / 255) * (canvas.height - 8);
                              ctx2d.fillStyle = '#c084fc';
                              ctx2d.fillRect(i * (barW + 2), canvas.height - h, barW, h);
                            }}
                          }}
                          draw();
                        }} catch (err) {{
                          status.textContent = '⚠️ Microphone access unavailable in this browser frame: ' + err.message;
                        }}
                      }};
                    }})();
                    </script>
                    """,
                    height=165,
                )

            # Room Live Stream Feed
            st.markdown('<div class="section-header">🔊 Room Voice Stream</div>', unsafe_allow_html=True)
            transmissions = list_room_transmissions(current_room.room_id)
            if not transmissions:
                st.caption("No voice transmissions in this room yet. Use Push-to-Talk above to speak!")
            else:
                for idx, msg in enumerate(transmissions):
                    eff_k = msg.get("effect", "helium")
                    eff_m = EFFECTS_REGISTRY.get(eff_k, EFFECTS_REGISTRY["helium"])
                    sender = msg.get("author", "Anonymous")
                    ts = msg.get("created_at", "")[11:19]
                    dur = msg.get("duration_sec", "")

                    st.markdown(f"""
                    <div class="feed-item" style="padding: 0.75rem 1rem;">
                      <div class="feed-header">
                        <div class="feed-avatar">{sender[:2].upper()}</div>
                        <div class="feed-meta">
                          <strong>{sender}</strong>
                          <small>{ts} UTC · {dur}s</small>
                        </div>
                        <div class="feed-effect-tag" style="color: {eff_m['color']}; border-color: {eff_m['color']}55;">
                          {eff_m['emoji']} {eff_m['label'].split(' ', 1)[-1]}
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    c_path = msg.get("path")
                    if c_path and Path(c_path).exists():
                        st.audio(Path(c_path).read_bytes(), format="audio/wav", autoplay=(idx == 0))


# ---------------------------------------------------------------------------
# ─────────────────────────── PAGE: ABOUT ───────────────────────────────────
# ---------------------------------------------------------------------------

def page_about():
    st.markdown("""
    <div class="vm-hero">
      <h1>🎃 Hacktoberfest — Contributor Guide</h1>
      <p>Welcome to VoiceMorph! All 12 DSP effects, Group Feeds, and Live Call rooms are ready.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="vm-card">
          <div class="section-header" style="margin-top: 0;">🚀 Quick Start</div>
          <pre style="font-family: monospace; font-size: 0.82rem; color: #94a3b8;
                      line-height: 1.7; background: #0a0c16; padding: 1rem;
                      border-radius: 8px; margin: 0; overflow-x: auto;">
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run main.py</pre>
        </div>

        <div class="vm-card" style="margin-top: 1rem;">
          <div class="section-header" style="margin-top: 0;">🎛️ Included 12 DSP Voice Effects</div>
          <p style="font-size: 0.86rem; color: #94a3b8; line-height: 1.6;">
            • <strong>🎈 Helium</strong> — WSOLA pitch shift up + highpass filter<br>
            • <strong>👹 Monster</strong> — Pitch shift down + bass boost + tanh growl<br>
            • <strong>🤖 Robotic</strong> — Dual-harmonic ring modulation<br>
            • <strong>🔊 Echo</strong> — Multi-tap delay line with adjustable decay<br>
            • <strong>🐿️ Chipmunk</strong> — Octave pitch shift + speed multiplier<br>
            • <strong>🏔️ Cave</strong> — Schroeder parallel comb-filter reverb<br>
            • <strong>🌊 Tremolo</strong> — Sinusoidal LFO amplitude modulation<br>
            • <strong>🎸 Vibrato</strong> — Fractional delay line LFO pitch warble<br>
            • <strong>📞 Telephone</strong> — 300–3200 Hz bandpass + warm line saturation<br>
            • <strong>🫧 Underwater</strong> — Submerged lowpass + bubbly chorus LFO<br>
            • <strong>🎶 Chorus</strong> — Multi-voice detuned delay ensemble<br>
            • <strong>🌀 Flanger</strong> — Swept short comb-filter jet phase effect
          </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="vm-card">
          <div class="section-header" style="margin-top: 0;">📁 Project Structure</div>
          <pre style="font-family: monospace; font-size: 0.82rem; color: #94a3b8;
                      line-height: 1.8; background: #0a0c16; padding: 1rem;
                      border-radius: 8px; margin: 0;">
voicemorph/
├── main.py          ← App router, Studio, Groups & Live Call UI
├── effects.py       ← 🎛️ 12 DSP effects + parameter schemas
├── audio_utils.py   ← Audio I/O, sample voice synth & waveforms
├── storage.py       ← Persistent JSON state for groups & rooms
├── live_call.py     ← 📡 WebSocket server & room voice broadcast
├── requirements.txt ← Python dependencies
└── data/            ← Auto-created at runtime
    ├── groups.json
    ├── rooms.json
    ├── clips/
    ├── feed/
    └── room_messages/</pre>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# ROUTER
# ---------------------------------------------------------------------------

page = st.session_state.page

if page == "studio":
    page_studio()
elif page == "groups":
    page_groups()
elif page == "live":
    page_live()
elif page == "about":
    page_about()

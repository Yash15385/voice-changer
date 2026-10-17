"""
storage.py — Local File-Based State Manager
============================================
🎙️ Funny Voice Changer — Hacktoberfest Open Source Project
------------------------------------------------------------
Handles all persistent state using JSON files and local directories.
No database setup required — perfect for local development and contribution.

Directory layout (created automatically on first run):
    data/
    ├── groups.json          ← Group definitions
    ├── rooms.json           ← Active live call rooms & participants
    ├── clips/               ← All processed audio clips (.wav)
    │   └── <clip_id>.wav
    ├── feed/                ← Per-group JSON feed files
    │   └── <group_id>.json
    └── room_messages/       ← Per-room live voice transmissions
        └── <room_id>.json
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# PATH CONSTANTS (anchored to project directory)
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CLIPS_DIR = DATA_DIR / "clips"
FEED_DIR = DATA_DIR / "feed"
ROOM_MESSAGES_DIR = DATA_DIR / "room_messages"
GROUPS_FILE = DATA_DIR / "groups.json"
ROOMS_FILE = DATA_DIR / "rooms.json"
TEMP_DIR = DATA_DIR / "temp"  # Ephemeral uploads — auto-cleaned

MAX_CLIP_AGE_HOURS = 24       # Clips older than this are eligible for cleanup
MAX_CLIPS_PER_GROUP = 20      # Rolling window per group feed
MAX_MESSAGES_PER_ROOM = 30    # Rolling window per live call room


# ---------------------------------------------------------------------------
# INITIALISATION
# ---------------------------------------------------------------------------

def init_storage() -> None:
    """Create all required directories and seed default groups if needed."""
    for d in [DATA_DIR, CLIPS_DIR, FEED_DIR, ROOM_MESSAGES_DIR, TEMP_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    if not GROUPS_FILE.exists():
        default_groups = [
            {
                "id": "hacktoberfest",
                "name": "🎃 Hacktoberfest Squad",
                "description": "The official open-source party room!",
                "emoji": "🎃",
                "color": "#f97316",
                "member_count": 42,
                "created_at": _now_iso(),
            },
            {
                "id": "memes",
                "name": "😂 Meme Lords",
                "description": "Only the dankest voice clips allowed.",
                "emoji": "😂",
                "color": "#eab308",
                "member_count": 17,
                "created_at": _now_iso(),
            },
            {
                "id": "devs",
                "name": "💻 Dev Cave",
                "description": "Ship code, share voice clips, repeat.",
                "emoji": "💻",
                "color": "#6366f1",
                "member_count": 88,
                "created_at": _now_iso(),
            },
            {
                "id": "gaming",
                "name": "🎮 Gaming Guild",
                "description": "GG EZ — voice-modded team comms.",
                "emoji": "🎮",
                "color": "#22c55e",
                "member_count": 31,
                "created_at": _now_iso(),
            },
        ]
        _write_json(GROUPS_FILE, default_groups)

    if not ROOMS_FILE.exists():
        _write_json(ROOMS_FILE, {})


# ---------------------------------------------------------------------------
# GROUPS
# ---------------------------------------------------------------------------

def get_groups() -> list[dict]:
    """Return all group definitions."""
    init_storage()
    return _read_json(GROUPS_FILE, default=[])


def get_group(group_id: str) -> Optional[dict]:
    """Return a single group by ID, or None."""
    for g in get_groups():
        if g["id"] == group_id:
            return g
    return None


def create_group(name: str, description: str = "", emoji: str = "🔊") -> dict:
    """Create and persist a new group."""
    groups = get_groups()
    slug = "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")[:20] or "group"
    group_id = f"{slug}_{uuid.uuid4().hex[:6]}"
    group = {
        "id": group_id,
        "name": name,
        "description": description or "Custom friend squad",
        "emoji": emoji or "🔊",
        "color": "#6366f1",
        "member_count": 1,
        "created_at": _now_iso(),
    }
    groups.append(group)
    _write_json(GROUPS_FILE, groups)
    return group


# ---------------------------------------------------------------------------
# CLIPS
# ---------------------------------------------------------------------------

def save_clip(
    audio_bytes: bytes,
    effect: str,
    author: str,
    original_filename: str,
) -> dict:
    """
    Persist a processed audio clip and return its metadata dict.
    """
    init_storage()
    clip_id = uuid.uuid4().hex
    clip_path = CLIPS_DIR / f"{clip_id}.wav"
    clip_path.write_bytes(audio_bytes)

    meta = {
        "id": clip_id,
        "path": str(clip_path),
        "effect": effect,
        "author": author,
        "original_filename": original_filename,
        "created_at": _now_iso(),
        "size_bytes": len(audio_bytes),
    }
    return meta


def get_clip_path(clip_id: str) -> Optional[Path]:
    """Return Path to a clip file, or None if not found."""
    p = CLIPS_DIR / f"{clip_id}.wav"
    return p if p.exists() else None


# ---------------------------------------------------------------------------
# GROUP FEED
# ---------------------------------------------------------------------------

def post_to_group(group_id: str, clip_meta: dict) -> None:
    """Append a clip to a group's feed (rolling window)."""
    init_storage()
    feed_file = FEED_DIR / f"{group_id}.json"
    feed = _read_json(feed_file, default=[])
    feed.append(clip_meta)
    if len(feed) > MAX_CLIPS_PER_GROUP:
        feed = feed[-MAX_CLIPS_PER_GROUP:]
    _write_json(feed_file, feed)


def get_group_feed(group_id: str) -> list[dict]:
    """Return feed for a group, newest-first."""
    init_storage()
    feed_file = FEED_DIR / f"{group_id}.json"
    feed = _read_json(feed_file, default=[])
    return list(reversed(feed))


def delete_post(group_id: str, clip_id: str) -> bool:
    """Remove a specific post from a group's feed."""
    feed_file = FEED_DIR / f"{group_id}.json"
    feed = _read_json(feed_file, default=[])
    new_feed = [c for c in feed if c.get("id") != clip_id]
    if len(new_feed) == len(feed):
        return False
    _write_json(feed_file, new_feed)
    return True


# ---------------------------------------------------------------------------
# LIVE CALL ROOM PERSISTENCE
# ---------------------------------------------------------------------------

def load_rooms_store() -> dict[str, dict]:
    """Load persisted rooms dictionary from disk."""
    init_storage()
    data = _read_json(ROOMS_FILE, default={})
    return data if isinstance(data, dict) else {}


def save_rooms_store(rooms: dict[str, dict]) -> None:
    """Persist rooms dictionary to disk."""
    init_storage()
    _write_json(ROOMS_FILE, rooms)


def post_room_message(room_id: str, clip_meta: dict) -> None:
    """Append a live voice message to a call room's stream log."""
    init_storage()
    msg_file = ROOM_MESSAGES_DIR / f"{room_id}.json"
    messages = _read_json(msg_file, default=[])
    messages.append(clip_meta)
    if len(messages) > MAX_MESSAGES_PER_ROOM:
        messages = messages[-MAX_MESSAGES_PER_ROOM:]
    _write_json(msg_file, messages)


def get_room_messages(room_id: str) -> list[dict]:
    """Return voice messages for a live call room, newest-first."""
    init_storage()
    msg_file = ROOM_MESSAGES_DIR / f"{room_id}.json"
    messages = _read_json(msg_file, default=[])
    return list(reversed(messages))


# ---------------------------------------------------------------------------
# CLEANUP
# ---------------------------------------------------------------------------

def cleanup_temp() -> int:
    """Delete all files in the temp directory. Returns count deleted."""
    init_storage()
    count = 0
    for f in TEMP_DIR.glob("*"):
        try:
            f.unlink()
            count += 1
        except OSError:
            pass
    return count


def cleanup_old_clips(max_age_hours: int = MAX_CLIP_AGE_HOURS) -> int:
    """
    Delete clip files older than `max_age_hours`.
    Also removes orphaned clip IDs from all group feeds.
    """
    init_storage()
    cutoff = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
    deleted_ids: set[str] = set()
    count = 0

    for clip_file in CLIPS_DIR.glob("*.wav"):
        if clip_file.stat().st_mtime < cutoff:
            clip_id = clip_file.stem
            try:
                clip_file.unlink()
                deleted_ids.add(clip_id)
                count += 1
            except OSError:
                pass

    if deleted_ids:
        for feed_file in FEED_DIR.glob("*.json"):
            feed = _read_json(feed_file, default=[])
            new_feed = [c for c in feed if c.get("id") not in deleted_ids]
            if len(new_feed) != len(feed):
                _write_json(feed_file, new_feed)

    return count


def get_storage_stats() -> dict:
    """Return storage usage statistics."""
    init_storage()
    clips = list(CLIPS_DIR.glob("*.wav"))
    total_bytes = sum(c.stat().st_size for c in clips if c.exists())
    return {
        "clip_count": len(clips),
        "total_mb": round(total_bytes / (1024 * 1024), 2),
        "groups": len(get_groups()),
    }


# ---------------------------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default=None):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return default if default is not None else {}


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

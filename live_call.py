"""
live_call.py — Real-Time WebSocket & Live Call Room Engine
===========================================================
🎙️ Funny Voice Changer — Hacktoberfest Open Source Project
-----------------------------------------------------------
Provides:
  1. Persistent multi-user CallRoom state (backed by storage.py so multiple
     tabs/users see the same rooms, participants, and active voice effects).
  2. Real-time DSP stream chunk processing (`process_stream_chunk`) and
     push-to-talk / live voice clip broadcasting (`broadcast_voice_to_room`).
  3. An asynchronous WebSocket audio relay server (`LiveCallWebSocketServer`)
     that modulates incoming 16-bit PCM chunks per room and broadcasts them
     to all connected room peers.
"""

import asyncio
import json
import socket
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from audio_utils import (
    TARGET_SAMPLE_RATE,
    audio_to_wav_bytes,
    get_audio_info,
    load_audio,
    resample_audio,
    trim_silence,
)
from effects import EFFECTS_REGISTRY, apply_effect
from storage import (
    get_room_messages,
    load_rooms_store,
    post_room_message,
    save_clip,
    save_rooms_store,
)

# ---------------------------------------------------------------------------
# IN-MEMORY CACHE + PERSISTENT ROOM STORE SYNC
# ---------------------------------------------------------------------------

_active_rooms: dict[str, "CallRoom"] = {}
DEFAULT_WS_HOST = "localhost"
DEFAULT_WS_PORT = 8765


class CallRoom:
    """
    Represents a live voice-modulated call room.
    Synchronizes automatically with `data/rooms.json` for multi-tab support.
    """

    def __init__(
        self,
        group_id: str,
        host: str,
        default_effect: str = "helium",
        room_id: Optional[str] = None,
        created_at: Optional[str] = None,
        participants: Optional[list[dict]] = None,
        is_active: bool = True,
        status: str = "live",
    ):
        self.room_id = room_id or f"room_{uuid.uuid4().hex[:8]}"
        self.group_id = group_id
        self.host = host
        self.default_effect = default_effect if default_effect in EFFECTS_REGISTRY else "helium"
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.participants: list[dict] = participants if participants is not None else [
            {
                "name": host,
                "joined_at": self.created_at,
                "is_host": True,
                "effect": self.default_effect,
            }
        ]
        self.is_active = is_active
        self.status = status

    def to_dict(self) -> dict:
        return {
            "room_id": self.room_id,
            "group_id": self.group_id,
            "host": self.host,
            "default_effect": self.default_effect,
            "created_at": self.created_at,
            "participants": self.participants,
            "participant_count": len(self.participants),
            "is_active": self.is_active,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CallRoom":
        return cls(
            group_id=data.get("group_id", "hacktoberfest"),
            host=data.get("host", "Anonymous"),
            default_effect=data.get("default_effect", "helium"),
            room_id=data.get("room_id"),
            created_at=data.get("created_at"),
            participants=data.get("participants", []),
            is_active=data.get("is_active", True),
            status=data.get("status", "live"),
        )

    def add_participant(self, name: str, effect: Optional[str] = None) -> dict:
        chosen_effect = effect or self.default_effect
        for p in self.participants:
            if p["name"].lower() == name.lower():
                if effect and effect in EFFECTS_REGISTRY:
                    p["effect"] = effect
                _persist_room(self)
                return p

        participant = {
            "name": name,
            "joined_at": datetime.now(timezone.utc).isoformat(),
            "is_host": False,
            "effect": chosen_effect if chosen_effect in EFFECTS_REGISTRY else self.default_effect,
        }
        self.participants.append(participant)
        self.status = "live"
        _persist_room(self)
        return participant

    def set_participant_effect(self, name: str, effect_key: str) -> None:
        if effect_key not in EFFECTS_REGISTRY:
            return
        found = False
        for p in self.participants:
            if p["name"].lower() == name.lower():
                p["effect"] = effect_key
                found = True
        if not found:
            self.add_participant(name, effect=effect_key)
        else:
            _persist_room(self)

    def get_participant_effect(self, name: str) -> str:
        for p in self.participants:
            if p["name"].lower() == name.lower():
                return p.get("effect", self.default_effect)
        return self.default_effect

    def remove_participant(self, name: str) -> None:
        self.participants = [p for p in self.participants if p["name"].lower() != name.lower()]
        if len(self.participants) == 0:
            self.is_active = False
            self.status = "ended"
        _persist_room(self)


def _sync_from_disk() -> None:
    """Sync in-memory `_active_rooms` with disk store."""
    raw_rooms = load_rooms_store()
    for r_id, r_dict in raw_rooms.items():
        _active_rooms[r_id] = CallRoom.from_dict(r_dict)


def _persist_room(room: CallRoom) -> None:
    """Save a room's state to disk and memory."""
    _active_rooms[room.room_id] = room
    raw_rooms = load_rooms_store()
    raw_rooms[room.room_id] = room.to_dict()
    save_rooms_store(raw_rooms)


# ---------------------------------------------------------------------------
# ROOM MANAGEMENT API
# ---------------------------------------------------------------------------

def create_room(group_id: str, host: str, default_effect: str = "helium") -> CallRoom:
    """Create and register a new live call room."""
    _sync_from_disk()
    room = CallRoom(group_id=group_id, host=host, default_effect=default_effect)
    _persist_room(room)
    return room


def get_room(room_id: str) -> Optional[CallRoom]:
    """Fetch an active room by ID."""
    _sync_from_disk()
    room = _active_rooms.get(room_id)
    if room and room.is_active:
        return room
    return None


def get_active_rooms(group_id: Optional[str] = None) -> list[dict]:
    """Return all active rooms, optionally filtered by group."""
    _sync_from_disk()
    rooms = [r for r in _active_rooms.values() if r.is_active]
    if group_id:
        rooms = [r for r in rooms if r.group_id == group_id]
    return [r.to_dict() for r in rooms]


def end_room(room_id: str) -> bool:
    """Mark a room as ended."""
    _sync_from_disk()
    room = _active_rooms.get(room_id)
    if not room:
        return False
    room.is_active = False
    room.status = "ended"
    _persist_room(room)
    return True


# ---------------------------------------------------------------------------
# LIVE VOICE TRANSMISSION & STREAM CHUNK PROCESSOR
# ---------------------------------------------------------------------------

def process_stream_chunk(
    chunk: bytes,
    effect_key: str,
    sample_rate: int = TARGET_SAMPLE_RATE,
    chunk_size: int = 1024,
) -> bytes:
    """
    Process a raw 16-bit PCM audio chunk through the specified DSP voice effect.
    """
    if not chunk:
        return b""
    # Ensure buffer length is a multiple of 2 bytes (int16)
    if len(chunk) % 2 != 0:
        chunk = chunk[:-1]
    if not chunk:
        return b""

    audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
    if len(audio) == 0:
        return b""

    effect = effect_key if effect_key in EFFECTS_REGISTRY else "helium"
    processed = apply_effect(effect, audio, sample_rate)

    # Match original chunk length for seamless streaming playback if desired
    if len(processed) > len(audio) and effect not in ("echo", "cave"):
        processed = processed[: len(audio)]

    pcm = (np.clip(processed, -1.0, 1.0) * 32767).astype(np.int16)
    return pcm.tobytes()


def broadcast_voice_to_room(
    room_id: str,
    raw_audio_bytes: bytes,
    sender: str,
    effect_key: Optional[str] = None,
    filename: str = "live_mic.wav",
) -> dict:
    """
    Modulate a recorded or synthesized audio transmission with the participant's
    selected room voice effect and broadcast it to the room's live stream feed.
    """
    room = get_room(room_id)
    if not room:
        raise ValueError(f"Room '{room_id}' is not active.")

    chosen_effect = effect_key or room.get_participant_effect(sender)
    if chosen_effect not in EFFECTS_REGISTRY:
        chosen_effect = room.default_effect

    room.add_participant(sender, effect=chosen_effect)

    audio, sr = load_audio(raw_audio_bytes, filename)
    if sr != TARGET_SAMPLE_RATE:
        audio = resample_audio(audio, sr, TARGET_SAMPLE_RATE)
        sr = TARGET_SAMPLE_RATE

    audio = trim_silence(audio, sample_rate=sr)
    processed = apply_effect(chosen_effect, audio, sr)
    wav_bytes = audio_to_wav_bytes(processed, sr)
    info = get_audio_info(processed, sr)

    clip_meta = save_clip(
        audio_bytes=wav_bytes,
        effect=chosen_effect,
        author=sender,
        original_filename=filename,
    )
    clip_meta["room_id"] = room_id
    clip_meta["duration_sec"] = info["duration_sec"]
    post_room_message(room_id, clip_meta)
    return clip_meta


def list_room_transmissions(room_id: str) -> list[dict]:
    """Return all modulated voice transmissions in a room (newest first)."""
    return get_room_messages(room_id)


# ---------------------------------------------------------------------------
# BACKGROUND WEBSOCKET AUDIO RELAY SERVER
# ---------------------------------------------------------------------------

_ws_server_thread: Optional[threading.Thread] = None
_ws_server_port: Optional[int] = None
_ws_connected_clients: dict[str, set] = {}


def _is_port_in_use(port: int, host: str = DEFAULT_WS_HOST) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex((host, port)) == 0


def ensure_websocket_server(port: int = DEFAULT_WS_PORT) -> dict:
    """
    Start the background asyncio WebSocket relay server if `websockets` is
    installed and the server is not already running.
    """
    global _ws_server_thread, _ws_server_port

    if _ws_server_port and _is_port_in_use(_ws_server_port):
        return {"running": True, "port": _ws_server_port, "engine": "websockets"}

    try:
        import websockets  # noqa: F401
    except ImportError:
        return {
            "running": True,
            "port": port,
            "engine": "in-app-transceiver",
            "note": "Running built-in room transceiver (install `websockets` for raw WS socket relay).",
        }

    # Find an available port starting at `port`
    chosen_port = port
    for candidate in range(port, port + 10):
        if not _is_port_in_use(candidate):
            chosen_port = candidate
            break

    async def _ws_handler(websocket):
        path = getattr(websocket, "path", "") or "/"
        parts = [p for p in path.strip("/").split("/") if p]
        room_id = parts[0] if parts else "default_room"
        effect_key = parts[1] if len(parts) > 1 else "helium"

        peers = _ws_connected_clients.setdefault(room_id, set())
        peers.add(websocket)
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    room = get_room(room_id)
                    eff = room.default_effect if room else effect_key
                    modulated = process_stream_chunk(message, eff)
                    dead = []
                    for peer in list(peers):
                        try:
                            await peer.send(modulated)
                        except Exception:
                            dead.append(peer)
                    for d in dead:
                        peers.discard(d)
                elif isinstance(message, str):
                    try:
                        payload = json.loads(message)
                        if payload.get("type") == "set_effect":
                            effect_key = payload.get("effect", effect_key)
                            await websocket.send(
                                json.dumps({"type": "ack", "effect": effect_key, "room_id": room_id})
                            )
                    except json.JSONDecodeError:
                        pass
        finally:
            peers.discard(websocket)

    def _run_loop(bind_port: int):
        import websockets

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _serve():
            async with websockets.serve(_ws_handler, DEFAULT_WS_HOST, bind_port):
                await asyncio.Future()

        try:
            loop.run_until_complete(_serve())
        except Exception:
            pass

    _ws_server_port = chosen_port
    _ws_server_thread = threading.Thread(target=_run_loop, args=(chosen_port,), daemon=True)
    _ws_server_thread.start()
    return {"running": True, "port": chosen_port, "engine": "websockets"}


class WebSocketStub:
    """
    Backward-compatible helper providing live connection metadata and
    automatic stream relay handling.
    """

    @staticmethod
    async def echo_handler(data: bytes, effect_key: str = "helium") -> bytes:
        """Apply real-time DSP effect to an incoming PCM chunk."""
        await asyncio.sleep(0)
        return process_stream_chunk(data, effect_key=effect_key)

    @staticmethod
    def get_connection_info(room_id: str) -> dict:
        """Return live connection info for the specified room."""
        srv = ensure_websocket_server()
        port = srv.get("port", DEFAULT_WS_PORT)
        room = get_room(room_id)
        default_effect = room.default_effect if room else "helium"
        return {
            "room_id": room_id,
            "ws_url": f"ws://{DEFAULT_WS_HOST}:{port}/{room_id}/{default_effect}",
            "stun_server": "stun:stun.l.google.com:19302",
            "engine": srv.get("engine", "in-app-transceiver"),
            "status": "active",
            "default_effect": default_effect,
            "participants": len(room.participants) if room else 0,
        }


if __name__ == "__main__":
    info = ensure_websocket_server(DEFAULT_WS_PORT)
    print(f"🎙️ VoiceMorph Live Call Server active: {info}")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("Shutting down.")

import asyncio
from unittest.mock import MagicMock

import numpy as np


def test_live_silence_timeout_is_opt_in(monkeypatch):
    import app.api.websocket as ws_mod

    asr = MagicMock()
    speaker = MagicMock()

    class DummyWebSocket:
        def __init__(self):
            self._engine_snapshot = ws_mod.EngineSnapshot(asr=asr, speaker=speaker)
            self.sent = []

        async def send_json(self, msg):
            self.sent.append(msg)

    websocket = DummyWebSocket()
    queue = asyncio.Queue()
    queue.put_nowait((0, np.zeros(1600, dtype=np.int16).tobytes()))
    stop_event = asyncio.Event()

    monkeypatch.setattr(ws_mod.config.audio, "skip_frame_threshold", 100)
    monkeypatch.setattr(ws_mod.config.audio, "timeout_seconds", 1)
    monkeypatch.setattr(ws_mod.config.audio, "live_auto_stop_on_silence", False, raising=False)
    monkeypatch.setattr(ws_mod.config.audio, "max_segment_seconds", 5)

    clock_values = iter([0.0, 100.0])
    monkeypatch.setattr(ws_mod.time, "time", lambda: next(clock_values, 100.0))

    async def run_processor():
        task = asyncio.create_task(
            ws_mod.audio_processor(websocket, queue, "test_user", stop_event)
        )
        while not queue.empty():
            await asyncio.sleep(0)
        stop_event.set()
        queue.put_nowait((1600, b""))
        await task

    asyncio.run(run_processor())

    assert not any(msg.get("text") == "LINK_IDLE_TIMEOUT" for msg in websocket.sent)


def test_live_silence_timeout_can_auto_stop_when_enabled(monkeypatch):
    import app.api.websocket as ws_mod

    asr = MagicMock()
    speaker = MagicMock()

    class DummyWebSocket:
        def __init__(self):
            self._engine_snapshot = ws_mod.EngineSnapshot(asr=asr, speaker=speaker)
            self.sent = []

        async def send_json(self, msg):
            self.sent.append(msg)

    websocket = DummyWebSocket()
    queue = asyncio.Queue()
    queue.put_nowait((0, np.zeros(1600, dtype=np.int16).tobytes()))
    stop_event = asyncio.Event()

    monkeypatch.setattr(ws_mod.config.audio, "skip_frame_threshold", 100)
    monkeypatch.setattr(ws_mod.config.audio, "timeout_seconds", 1)
    monkeypatch.setattr(ws_mod.config.audio, "live_auto_stop_on_silence", True, raising=False)
    monkeypatch.setattr(ws_mod.config.audio, "max_segment_seconds", 5)

    clock_values = iter([0.0, 100.0])
    monkeypatch.setattr(ws_mod.time, "time", lambda: next(clock_values, 100.0))

    asyncio.run(ws_mod.audio_processor(websocket, queue, "test_user", stop_event))

    assert stop_event.is_set()
    assert any(msg.get("text") == "LINK_IDLE_TIMEOUT" for msg in websocket.sent)

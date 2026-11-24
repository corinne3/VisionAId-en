"""Phase 22 (integration): guidance sounds served to the phone and live sound instructions."""

from __future__ import annotations

import asyncio
import json
import math
import socket
import time

import aiohttp
import pytest

from visionaid.apps.live_blocks import LiveGuidanceBlock
from visionaid.bus.messages import Pose, SoundCommand
from visionaid.phone.certs import client_ssl_context
from visionaid.phone.protocol import PROTOCOL_VERSION
from visionaid.phone.server import PhoneServer
from visionaid.utils.config import load_config, load_raw

pytestmark = [pytest.mark.integration, pytest.mark.unit]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def cfg(tmp_path):
    return load_config(
        overrides={"phone.certs_dir": str(tmp_path / "certs"), "phone.host": "127.0.0.1"}
    )


async def _client(srv, cmd: SoundCommand):
    base = f"https://localhost:{srv.https_port}"
    ssl = client_ssl_context(srv.certs)
    got: dict = {}
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(f"{base}/ws", ssl=ssl) as ws:
            await ws.send_str(
                json.dumps(
                    {"type": "hello", "protocol": PROTOCOL_VERSION, "ua": "test", "t_phone": 0.0}
                )
            )
            got["welcome"] = json.loads((await ws.receive(timeout=5)).data)
            srv.send_sound(cmd)  # from a thread other than the server's, like the source block
            t_end = time.time() + 5
            while time.time() < t_end:
                m = json.loads((await ws.receive(timeout=5)).data)
                if m["type"] == "sound":
                    got["sound"] = m
                    break
        for k, url in got["welcome"]["audio"]["sounds"].items():
            async with s.get(base + url, ssl=ssl) as r:
                got[f"http_{k}"] = (r.status, len(await r.read()))
    return got


def test_welcome_audio_sounds_served_and_sound_messages(cfg):
    cmd = SoundCommand(2.0, 7, "NAVIGATING", "guide", 0.5, -0.6, 3.0, 1.0, 0.0, -14.0, "")
    with PhoneServer(cfg, https_port=free_port(), http_port=0) as srv:
        got = asyncio.run(_client(srv, cmd))
    a = got["welcome"]["audio"]
    assert a["sounds"]["guide"].endswith("chemin.wav")
    assert a["sounds"]["obstacle"].endswith("obstacles.wav")
    assert a["obstacle_smoothing_s"] == pytest.approx(cfg.audio.obstacle_smoothing_s)
    for k in ("guide", "search", "arrived", "obstacle"):
        status, size = got[f"http_{k}"]
        assert status == 200 and size > 10_000
    s = got["sound"]
    assert s["mode"] == "guide" and s["pan"] == pytest.approx(-0.6) and s["interval"] == 1.0
    assert s["obstacle_db"] == pytest.approx(-14.0)


def _pose(t, idx, quality="ok"):
    import numpy as np

    T = np.eye(4)
    T[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]  # camera: optical axis towards world +x
    T[2, 3] = 1.3
    return Pose(t, idx, T, quality)


def test_live_guidance_block_sound_and_stale_pose():
    blk = LiveGuidanceBlock(load_raw(), stale_s=0.2)
    assert blk.process({}) is None  # no pose: silence
    out = blk.process({"pose_world": _pose(0.0, 1)})
    assert isinstance(out["sound"], SoundCommand) and out["sound"].mode == "search"
    time.sleep(0.6)
    for _ in range(3):  # frozen pose: tracking lost → "stop"
        out = blk.process({"pose_world": _pose(0.0, 1)})
        time.sleep(0.2)
    assert out["cue"].state == "TRACKING_LOST" and out["sound"].mode == "lost"
    assert not math.isnan(out["sound"].interval_s)

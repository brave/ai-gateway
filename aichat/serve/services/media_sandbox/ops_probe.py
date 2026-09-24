from __future__ import annotations

import os
import time


def op_probe_net(**_ignored) -> dict:
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    try:
        s.connect(("1.1.1.1", 80))
    finally:
        s.close()
    return {"connected": True}


def op_probe_env(**_ignored) -> dict:
    return {"env_keys": sorted(os.environ.keys())}


def op_spin_cpu(seconds: float = 1, **_ignored) -> dict:
    duration = min(float(seconds), 60)
    deadline = time.monotonic() + duration
    x = 0
    while time.monotonic() < deadline:
        for _ in range(100_000):
            x = (x + 1) % 100_003
    return {"spun": duration, "x": x}


def op_probe_fs_write(**_ignored) -> dict:
    import platform

    # Probe purpose: verify Landlock blocks /etc writes on Linux. On other
    # platforms (no Landlock) skip to avoid polluting the filesystem.
    if platform.system() != "Linux":
        return {"write_denied": None, "skipped": "non-linux"}
    try:
        with open("/etc/media-sandbox-probe", "w") as f:
            f.write("x")
    except PermissionError as e:
        return {"write_denied": True, "err": str(e)}
    os.remove("/etc/media-sandbox-probe")
    return {"write_denied": False}


PROBE_HANDLERS = {
    "probe_net": op_probe_net,
    "probe_env": op_probe_env,
    "probe_fs_write": op_probe_fs_write,
    "spin_cpu": op_spin_cpu,
}

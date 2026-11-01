"""HD-map LiDAR range-map control for an exported RDS-HQ clip (framework ``cosmos-auto-lidar-rangemap``).

The control is geometry only: map layers + object cuboids ray-cast onto the Pandar128-style 128 x 1800 lattice
using sweep poses from the exported ``vehicle_pose`` tar and the nominal top-LiDAR mount of the rig
(``--allow-default-extrinsic``; the Hyperion asset has no LiDAR entry). 297 camera frames at 30 fps = 99 sweeps at
10 Hz, which needs 298 pose frames (export with ``pad_frames=1``). No recorded returns and no model weights are used;
intensity in ``lidar_control.safetensors`` is a placeholder (1.0).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

DEFAULT_PY = "~/tsf/venvs/wsm/bin/python"


def render_lidar(clip_root, clip_id: str, out_dir, *, max_sweeps: int | None = None, prefilter: bool = False,
                 device: str = "cpu", python: str | None = None, camera_name: str = "camera_front_wide_120fov",
                 timeout: float | None = None) -> dict:
    """Run the range-map renderer; returns ``{outputs, sweeps, wall_s, ...}`` and writes ``lidar_timings.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    py = python or str(Path(os.environ.get("TSF_WSM_PYTHON", DEFAULT_PY)).expanduser())
    cmd = [py, "-m", "cosmos_auto.lidar.cli", "--input-root", str(clip_root), "--output-root", str(out_dir),
           "--clip-id", clip_id, "--camera-name", camera_name, "--allow-default-extrinsic", "--export-control",
           "--no-vis", "--device", device]
    if max_sweeps:
        cmd += ["--max-frames", str(max_sweeps)]
    if prefilter:
        cmd += ["--prefilter"]
    t = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    wall = time.perf_counter() - t
    (out_dir / "lidar.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"cosmos-auto-lidar-rangemap failed (rc={r.returncode}):\n{r.stderr[-2000:]}")
    res = {}
    for line in r.stdout.splitlines():
        if line.startswith("{") and '"clip_id"' in line:
            res = json.loads(line)
    res.update(wall_s=round(wall, 2), device=device, prefilter=prefilter, cmd=" ".join(cmd))
    (out_dir / "lidar_timings.json").write_text(json.dumps(res, indent=1))
    return res

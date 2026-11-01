#!/usr/bin/env python3
"""Host-side prep of ONE PhysicalAI-AV clip for the REFe sanity run (needs pandas + scipy; run in the TSF venv on thor3).

Reads (all under ``--root``, layout of the HF dataset ``nvidia/PhysicalAI-Autonomous-Vehicles``)
    labels/egomotion/egomotion.chunk_<c>.zip                    or  x/ego/<clip>.egomotion.parquet (already unzipped)
    calibration/camera_intrinsics/camera_intrinsics.chunk_<c>.parquet
    calibration/sensor_extrinsics.offline/sensor_extrinsics.offline.chunk_<c>.parquet
    camera/<cam>/<cam>.chunk_<c>.zip                            (mp4 + frame timestamps of the clip)
and writes to ``--out``: ``rig.json`` (hyperion8.json schema, this clip's own calibration), ``ego.npz`` (100 Hz egomotion of the
contiguous part of the clip), ``ts_<cam>.npy`` (frame timestamps, s) and the clip's mp4s.

PhysicalAI-AV is licensed internal-only: ``--out`` must stay on thor3 (it lives under ~/data/pai_av/derived). Nothing here is
meant to be copied off the machine.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

CAMS = ("camera_front_wide_120fov", "camera_cross_left_120fov", "camera_cross_right_120fov",
        "camera_rear_left_70fov", "camera_rear_right_70fov", "camera_rear_tele_30fov")


def nv12_to_bgr(buf: bytes, w: int, h: int) -> np.ndarray:
    """NV12 (BT.709, limited range; the PhysicalAI-AV HEVC streams carry no colour metadata) -> uint8 BGR."""
    a = np.frombuffer(buf, np.uint8)
    y = (a[: w * h].reshape(h, w).astype(np.float32) - 16.0) * 1.1644
    uv = cv2.resize(a[w * h:].reshape(h // 2, w // 2, 2).astype(np.float32) - 128.0, (w, h), interpolation=cv2.INTER_LINEAR)
    u, v = uv[..., 0], uv[..., 1]
    rgb = np.stack([y + 1.7927 * v, y - 0.2132 * u - 0.5329 * v, y + 2.1124 * u], -1)
    return np.clip(rgb[..., ::-1] + 0.5, 0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / "data/pai_av"))
    ap.add_argument("--clip", default="bcc52cd1-9507-425e-a5a6-5f8a795ed525")
    ap.add_argument("--chunk", type=int, default=2800)
    ap.add_argument("--out", required=True)
    ap.add_argument("--t-start", type=float, default=2.0, help="first sanity timestep (s)")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--src", default=str(Path(__file__).resolve().parents[2] / "src"), help="dir holding the tsf package")
    a = ap.parse_args()
    import pandas as pd
    from scipy.spatial.transform import Rotation

    sys.path.insert(0, a.src)
    from tsf.closedloop import rig_adapter as ra

    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    c4 = f"{a.chunk:04d}"

    rig = ra.pai_av_rig(a.clip, str(root / f"calibration/camera_intrinsics/camera_intrinsics.chunk_{c4}.parquet"),
                        str(root / f"calibration/sensor_extrinsics.offline/sensor_extrinsics.offline.chunk_{c4}.parquet"), CAMS)
    (out / "rig.json").write_text(json.dumps({n: {"ftheta_intrinsics": i.tolist(), "camera_to_vehicle": T.tolist()}
                                              for n, (i, T) in rig.items()}, indent=1))

    pq = root / f"x/ego/{a.clip}.egomotion.parquet"
    if pq.exists():
        e = pd.read_parquet(pq)
    else:
        with zipfile.ZipFile(root / f"labels/egomotion/egomotion.chunk_{c4}.zip") as z:
            e = pd.read_parquet(z.open(f"{a.clip}.egomotion.parquet"))
    t = e.timestamp.values / 1e6
    brk = np.where(np.diff(t) > 0.05)[0]                    # the file ends in a sparse tail of unrelated samples
    n = (brk[0] + 1) if len(brk) else len(t)
    e = e.iloc[:n]
    t = t[:n]
    yaw = np.unwrap(Rotation.from_quat(e[["qx", "qy", "qz", "qw"]].values).as_euler("zyx")[:, 0])
    np.savez(out / "ego.npz", t=t, x=e.x.values, y=e.y.values, yaw=yaw, vx=e.vx.values, vy=e.vy.values,
             ax=e.ax.values, ay=e.ay.values, curvature=e.curvature.values)

    for cam in CAMS:
        with zipfile.ZipFile(root / f"camera/{cam}/{cam}.chunk_{c4}.zip") as z:
            for name in (f"{a.clip}.{cam}.mp4", f"{a.clip}.{cam}.timestamps.parquet"):
                if not (out / name).exists():
                    z.extract(name, out)
            ts = pd.read_parquet(z.open(f"{a.clip}.{cam}.timestamps.parquet")).timestamp.values / 1e6
        np.save(out / f"ts_{cam}.npy", ts)
        # the clips are HEVC: decode the wanted frames with the host ffmpeg (the torch container's OpenCV has no HEVC decoder)
        idx = sorted({int(np.argmin(np.abs(ts - (a.t_start + a.step * k)))) for k in range(a.n)})
        fdir = out / "frames" / cam
        fdir.mkdir(parents=True, exist_ok=True)
        # accurate seek to the middle of frame i (constant 30 fps, pts(i) = i / 30). The host ffmpeg is a hardware (NVDEC) build: it
        # has no `select` / `scale` filter, so it emits raw NV12 and the BT.709 limited-range conversion is done here.
        for i in idx:
            raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{(i + 0.5) / 30.0:.5f}", "-i", str(out / f"{a.clip}.{cam}.mp4"),
                                  "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "nv12", "-"], check=True, capture_output=True).stdout
            cv2.imwrite(str(fdir / f"{i:05d}.png"), nv12_to_bgr(raw, 1920, 1080))
    print(f"prepared {a.clip}: ego {n} samples t=[{t[0]:.2f}, {t[-1]:.2f}] s, {len(CAMS)} cameras -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

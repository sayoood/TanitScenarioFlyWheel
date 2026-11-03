"""Runs *inside the WSM venv* (needs cosmos_auto); never imports ``tsf``.

Projects the sim's objects and traffic-light heads into every camera at the last frame of each chunk of a T1 run, with
the renderer's own F-theta model, so ``tsf.cosmos.adherence`` can compare the generated frames with the world they were
generated from. Writes ``<run>/adherence/proj.json``:

    {clip_id: {camera: {"size": [w, h],
                        "objects": [{"id", "type", "box": [u0, v0, u1, v1], "depth"}],
                        "heads":   [{"id", "state", "box", "depth", "facing"}]}}}

Boxes are in pixels of the renderer's native camera size; objects with a corner behind the camera are left out.

    ~/tsf/venvs/wsm/bin/python src/tsf/cosmos/_adherence_project.py runs/t1/bench/b480
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _box(cm, pts: np.ndarray, P: np.ndarray):
    xyd = np.asarray(cm.get_xy_and_depth(pts, P))
    if (xyd[:, 2] <= 0.5).any():
        return None, None
    u, v = xyd[:, 0], xyd[:, 1]
    if u.max() < 0 or v.max() < 0 or u.min() >= cm.width or v.min() >= cm.height:
        return None, None
    box = [max(0.0, u.min()), max(0.0, v.min()), min(cm.width - 1.0, u.max()), min(cm.height - 1.0, v.max())]
    if box[2] - box[0] < 1 or box[3] - box[1] < 1:
        return None, None
    return [round(float(x), 1) for x in box], round(float(xyd[:, 2].min()), 1)


def _cuboid(lwh, T) -> np.ndarray:
    l, w, h = lwh
    loc = np.array([[sx * l / 2, sy * w / 2, sz * h / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    T = np.asarray(T, float)
    return loc @ T[:3, :3].T + T[:3, 3]


def project_clip(rds: Path, clip: str) -> dict:
    from cosmos_auto.wm.camera.ftheta import FThetaCamera
    from cosmos_auto.wm.wds_utils import get_sample

    pose = get_sample(rds / "pose" / f"{clip}.tar")
    intr = get_sample(rds / "ftheta_intrinsic" / f"{clip}.tar")
    objs = get_sample(rds / "all_object_info" / f"{clip}.tar")
    cams = sorted(k.split(".")[1] for k in intr if k.startswith("ftheta_intrinsic."))
    last = max(int(k.split(".")[0]) for k in pose if k.endswith(f".{cams[0]}.npy"))
    frame = objs.get(f"{last:06d}.all_object_info.json", {})
    heads = []
    tl_tar = rds / "3d_traffic_lights" / f"{clip}.tar"
    if tl_tar.exists():
        st_tar = rds / "3d_traffic_lights_lightnet_status" / f"{clip}.tar"
        states = {}
        if st_tar.exists():
            agg = get_sample(st_tar)["aggregated_states.json"]["traffic_light_states"]
            states = {v["feature_id"]: v["state"][min(last, len(v["state"]) - 1)] for v in agg.values()}
        for lab in get_sample(tl_tar)["traffic_lights.json"]["labels"]:
            s3 = lab["labelData"]["shape3d"]
            fid = next(a["text"] for a in s3["attributes"] if a["name"] == "feature_id")
            vtx = np.asarray(s3["cuboid3d"]["vertices"], float)
            heads.append((fid, states.get(fid, "unknown"), vtx, vtx[3] - vtx[0]))   # local +x = the lit face's normal
    out = {}
    for cam in cams:
        cm = FThetaCamera.from_numpy(intr[f"ftheta_intrinsic.{cam}.npy"], device="cpu")
        P = pose[f"{last:06d}.pose.{cam}.npy"]
        rec = {"size": [int(cm.width), int(cm.height)], "frame": last, "objects": [], "heads": []}
        for oid, o in frame.items():
            box, depth = _box(cm, _cuboid(o["object_lwh"], o["object_to_world"]), P)
            if box:
                rec["objects"].append({"id": oid, "type": o["object_type"], "box": box, "depth": depth})
        for fid, state, vtx, normal in heads:
            box, depth = _box(cm, vtx, P)
            if box:
                facing = float(np.dot(normal[:2], (P[:3, 3] - vtx.mean(0))[:2])) > 0
                rec["heads"].append({"id": fid, "state": state, "box": box, "depth": depth, "facing": facing})
        out[cam] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="T1 run dir (holds rds/ and gen/)")
    ap.add_argument("--clip", default=None, help="only this clip")
    ap.add_argument("--stdout", action="store_true", help="print one JSON line instead of writing adherence/proj.json")
    a = ap.parse_args()
    run = Path(a.run)
    rds = run / "rds"
    clips = [a.clip] if a.clip else sorted(p.stem for p in (rds / "pose").glob("*.tar"))
    res = {c: project_clip(rds, c) for c in clips}
    if a.stdout:
        print(json.dumps(res), flush=True)
        return 0
    (run / "adherence").mkdir(exist_ok=True)
    (run / "adherence" / "proj.json").write_text(json.dumps(res))
    print(f"projected {len(res)} clips", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

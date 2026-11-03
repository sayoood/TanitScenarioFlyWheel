"""Adherence and quality score of a T1 run: do the generated cameras show the world they were generated from?

    signals  : lit colour seen at every signal head that faces the camera (projected box >= 8 px tall) against its state
    vehicles : a COCO detector's vehicles against the sim's projected vehicles: recall, and detections no sim vehicle
               explains (hallucinated vehicles per frame); optional, needs torch + torchvision
    quality  : sharpness and saturation of each chunk's last frame relative to chunk 0, and the seam ratio (mean frame
               change across a chunk hand-off over the mean change inside a chunk; 1 = invisible seam)

Signals and vehicles need ``adherence/proj.json`` from ``_adherence_project.py`` (WSM venv). Self-contained (numpy, cv2;
torch lazily) so it also runs as a plain script inside a PyTorch container:

    python src/tsf/cosmos/adherence.py runs/t1/bench/b480 [--vehicles] [--frames-dir DIR]
      -> runs/t1/bench/b480/adherence/score.json

``--frames-dir``: score enhanced frames (``DIR/<clip>/last_<camera>.png``) instead of the generated ones, e.g. a
Harmonizer post-pass, against the same projections.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np

MIN_HEAD_PX = 8              # head box height (generated pixels) below which the lit colour is not judged
MIN_VEH_PX = 14              # vehicle box height for recall / hallucination counting
VEH_TYPES = {"Car", "Truck"}
COCO_VEHICLES = {3: "car", 6: "bus", 8: "truck"}


def project_clip(run: Path, clip: str, python: str | None = None) -> dict:
    """``{camera: projection}`` of one exported clip, from ``_adherence_project.py`` in the WSM venv (``TSF_WSM_PYTHON``)."""
    py = python or os.path.expanduser(os.environ.get("TSF_WSM_PYTHON", "~/tsf/venvs/wsm/bin/python"))
    drv = Path(__file__).with_name("_adherence_project.py")
    out = subprocess.run([py, str(drv), str(run), "--clip", clip, "--stdout"], capture_output=True, text=True,
                         check=True).stdout
    return json.loads(out.strip().splitlines()[-1])[clip]


def _sharp(img: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def _sat(img: np.ndarray) -> float:
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].mean())


def lit_colours(crop: np.ndarray, max_blob: float) -> set[str]:
    """Colours of lamp-like blobs in a BGR crop: bright, saturated, at most ``max_blob`` px and not touching the crop
    border (a yellow bus or a teal van behind the head is large and runs out of the crop; a lit lamp does not).
    Cosmos draws lamps dimmer than a camera sees them (green arrows at V 80-125, red at 100-210), so red and green need
    less brightness than yellow, whose hue band is shared with the dry verge behind the heads (V 70-130)."""
    if crop.size == 0:
        return set()
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1], hsv[..., 2]
    masks = {"red": (v >= 100) & (s >= 100) & ((h <= 8) | (h >= 170)),
             "yellow": (v >= 170) & (s >= 100) & (h >= 12) & (h <= 35),
             "green": (v >= 75) & (s >= 80) & (h >= 55) & (h <= 90)}
    H, W = crop.shape[:2]
    found = set()
    for name, m in masks.items():
        n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        for x, y, w, hh, area in st[1:]:
            inside = x > 0 and y > 0 and x + w < W and y + hh < H
            if inside and 2 <= area <= max_blob:
                found.add(name)
                break
    return found


def _scale(box, sx, sy):
    return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]


def _iou(a, b) -> float:
    ix, iy = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def _site(fid: str) -> str:
    return fid[:-5] if fid.endswith("_left") else fid


def score_signals(img: np.ndarray, rec: dict) -> list[dict]:
    """One row per head site (main + left-arrow head) that faces the camera and is big enough to judge."""
    sx, sy = img.shape[1] / rec["size"][0], img.shape[0] / rec["size"][1]
    sites: dict[str, dict] = {}
    for h in rec["heads"]:
        if not h["facing"]:
            continue
        s = sites.setdefault(_site(h["id"]), {"boxes": [], "states": set()})
        s["boxes"].append(_scale(h["box"], sx, sy))
        s["states"].add(h["state"].split("_")[0])
    rows = []
    for sid, s in sites.items():
        b = np.array(s["boxes"])
        u0, v0, u1, v1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
        if v1 - v0 < MIN_HEAD_PX:
            continue
        pw, ph = 0.5 * (u1 - u0) + 2, 0.5 * (v1 - v0) + 2          # generated content wobbles by a few pixels
        crop = img[max(0, int(v0 - ph)): int(v1 + ph) + 1, max(0, int(u0 - pw)): int(u1 + pw) + 1]
        seen = lit_colours(crop, max_blob=0.35 * (u1 - u0) * (v1 - v0))
        expected = s["states"] - {"unknown"}
        verdict = "none" if not seen else "match" if seen <= expected else "wrong"
        rows.append({"site": sid, "expected": sorted(expected), "seen": sorted(seen), "verdict": verdict,
                     "px": round(float(v1 - v0), 1)})
    return rows


class VehicleScorer:
    def __init__(self, device: str | None = None, thresh: float = 0.6):
        import torch
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                                                box_score_thresh=thresh).eval().to(self.device)

    def __call__(self, img: np.ndarray, rec: dict) -> dict:
        t = self.torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)
        with self.torch.no_grad():
            det = self.model([t.to(self.device)])[0]
        dets = [b.tolist() for b, lab in zip(det["boxes"].cpu(), det["labels"].cpu()) if int(lab) in COCO_VEHICLES]
        sx, sy = img.shape[1] / rec["size"][0], img.shape[0] / rec["size"][1]
        sims = [_scale(o["box"], sx, sy) for o in rec["objects"] if o["type"] in VEH_TYPES]
        judged = [b for b in sims if b[3] - b[1] >= MIN_VEH_PX]
        used, hit = set(), 0
        for b in judged:
            best = max(range(len(dets)), key=lambda i: _iou(b, dets[i]), default=None)
            if best is not None and best not in used and _iou(b, dets[best]) >= 0.3:
                used.add(best)
                hit += 1
        ghosts = [d for d in dets if d[3] - d[1] >= MIN_VEH_PX and all(_iou(d, b) < 0.1 for b in sims)]
        return {"sim": len(judged), "found": hit, "detections": len(dets), "ghosts": len(ghosts)}


def _frames(video: str) -> list[np.ndarray]:
    cap, out = cv2.VideoCapture(video), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    return out


def score_run(run: Path, cams: list[str] | None = None, vehicles: bool = False, frames_dir: Path | None = None) -> dict:
    proj_p = run / "adherence" / "proj.json"
    proj = json.loads(proj_p.read_text()) if proj_p.exists() else {}
    clips = sorted(p.name for p in (run / "gen").iterdir()        # not the *_rejected / *_retry attempts of the judge
                   if re.fullmatch(r"t1_\d+", p.name) and (p / "result.json").exists())
    vs = VehicleScorer() if vehicles and proj else None
    rows, ref, prev_last = [], {}, {}
    seam, inner = {}, {}
    for clip in clips:
        r = json.loads((run / "gen" / clip / "result.json").read_text())
        for cam in cams or r["cameras"]:
            src = (frames_dir / clip / f"last_{cam}.png") if frames_dir else (run / "gen" / clip / f"last_{cam}.png")
            img = cv2.imread(str(src))
            if img is None:
                continue
            row = {"clip": clip, "cam": cam, "sharp": _sharp(img), "sat": _sat(img)}
            ref.setdefault(cam, row)
            row["sharp_rel"] = row["sharp"] / max(ref[cam]["sharp"], 1e-6)
            row["sat_rel"] = row["sat"] / max(ref[cam]["sat"], 1e-6)
            if not frames_dir:                       # seams: first new frame vs the previous chunk's last frame
                fr = _frames(r["videos"][cam])
                if len(fr) > 2:
                    if cam in prev_last:
                        seam.setdefault(cam, []).append(float(np.abs(fr[1].astype(float) - prev_last[cam]).mean()))
                    inner.setdefault(cam, []).extend(float(np.abs(fr[i + 1].astype(float) - fr[i]).mean())
                                                     for i in range(1, len(fr) - 1))
                    prev_last[cam] = fr[-1].astype(float)
            rec = proj.get(clip, {}).get(cam)
            if rec:
                row["signals"] = score_signals(img, rec)
                if vs is not None:
                    row["vehicles"] = vs(img, rec)
            rows.append(row)
    return {"run": str(run), "frames": str(frames_dir) if frames_dir else "generated", "rows": rows,
            "summary": summarize(rows, seam, inner)}


def summarize(rows: list[dict], seam: dict, inner: dict) -> dict:
    out = {}
    for cam in sorted({r["cam"] for r in rows}):
        rs = [r for r in rows if r["cam"] == cam]
        tail = rs[-max(1, len(rs) // 3):]
        sig = [s for r in rs for s in r.get("signals", [])]
        judged = [s for s in sig if s["verdict"] != "none"]
        veh = [r["vehicles"] for r in rs if "vehicles" in r]
        out[cam] = {
            "chunks": len(rs),
            "sharp_rel_mean": round(float(np.mean([r["sharp_rel"] for r in rs])), 3),
            "sharp_rel_last_third": round(float(np.mean([r["sharp_rel"] for r in tail])), 3),
            "sat_rel_last_third": round(float(np.mean([r["sat_rel"] for r in tail])), 3),
            "seam_ratio": round(float(np.mean(seam[cam]) / max(np.mean(inner[cam]), 1e-6)), 2) if seam.get(cam) else None,
            "signal_sites": len(sig), "signal_judged": len(judged),
            "signal_agreement": round(sum(s["verdict"] == "match" for s in judged) / len(judged), 3) if judged else None,
            "signal_wrong": sum(s["verdict"] == "wrong" for s in judged),
        }
        if veh:
            sim, found = sum(v["sim"] for v in veh), sum(v["found"] for v in veh)
            out[cam] |= {"vehicle_recall": round(found / sim, 3) if sim else None,
                         "ghost_vehicles_per_frame": round(sum(v["ghosts"] for v in veh) / len(veh), 2)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--cams", default=None, help="comma separated (default: the run's cameras)")
    ap.add_argument("--vehicles", action="store_true", help="also run the COCO vehicle detector (torch, torchvision)")
    ap.add_argument("--frames-dir", default=None, help="score DIR/<clip>/last_<camera>.png instead of the generated frames")
    ap.add_argument("--out", default=None, help="default: <run>/adherence/score[_<frames-dir name>].json")
    a = ap.parse_args()
    run = Path(a.run)
    fd = Path(a.frames_dir) if a.frames_dir else None
    res = score_run(run, a.cams.split(",") if a.cams else None, a.vehicles, fd)
    out = Path(a.out) if a.out else run / "adherence" / (f"score_{fd.name}.json" if fd else "score.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()

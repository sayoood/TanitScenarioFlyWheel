"""Evidence videos of one dense ViPE run (thor3): metric depth, NPC tracks in bird's-eye view, summary JSON.

    python render_clip.py RUN_DIR OUT_DIR [--no-video]

RUN_DIR = ``run_vipe(dense=True)`` output (w*/dense.npz, w*/clip.mp4, w*/chained.npz, track.npz); writes to OUT_DIR
    depth.mp4        RGB | MoGe-2 metric depth (fixed 0-80 m, sqrt scale, near = warm) | legend in metres
    bev.mp4          RGB | bird's-eye view in the ego frame (ego path, lanes of 3.5 m, NPC trails, ids, km/h)
    bev.png          whole-clip top-down plot in the planar start frame (time-coloured ego path, NPC tracks)
    npc_tracks.json  tracks; npc_summary.json  digest (what the UC2 request is built from)
Both videos and every derived file stay on thor3 (YouTube clips are CC-BY, PhysicalAI-AV is internal-only).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from tsf.ingest import npc, vipe

COLORS = {"car": (230, 140, 40), "truck": (40, 140, 240), "bus": (200, 80, 200), "motorcycle": (200, 200, 40),
          "bicycle": (60, 200, 60), "person": (60, 60, 230)}          # BGR
D_MAX = 80.0


def writer(path: Path, size: tuple[int, int], fps: float):
    import imageio_ffmpeg
    return subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
                             "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}", "-r", f"{fps:g}", "-i", "-", "-an",
                             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(path)],
                            stdin=subprocess.PIPE)


def frames(run: Path):
    """(video time, BGR frame, depth, instance) over the run, each window owning up to the middle of its overlap."""
    wins = sorted(run.glob("w*/dense.npz"))
    ts = [np.load(w.parent / "chained.npz")["t"] for w in wins]
    for i, w in enumerate(wins):
        lo = -np.inf if i == 0 else 0.5 * (ts[i][0] + ts[i - 1][-1])
        hi = np.inf if i + 1 == len(wins) else 0.5 * (ts[i + 1][0] + ts[i][-1])
        z = np.load(w)
        cap = cv2.VideoCapture(str(w.parent / "clip.mp4"))
        depth, inst = z["depth"], z["instance"]
        for k in range(len(ts[i])):
            ok, bgr = cap.read()
            if not ok:
                break
            if lo <= ts[i][k] < hi:
                yield float(ts[i][k]), bgr, depth[k].astype(np.float32), inst[k]
        cap.release()


def colorize(d: np.ndarray) -> np.ndarray:
    u = 1.0 - np.sqrt(np.clip(d, 0, D_MAX) / D_MAX)
    img = cv2.applyColorMap((u * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[d <= 0] = (40, 40, 40)
    return img


def legend(h: int, w: int = 86) -> np.ndarray:
    img = np.full((h, w, 3), 24, np.uint8)
    y0, y1 = 40, h - 24
    for yy in range(y0, y1):
        d = D_MAX * (1 - (yy - y0) / (y1 - y0)) ** 2
        img[yy, 8:28] = colorize(np.array([[max(d, 1e-3)]]))[0, 0]
    for m in (2, 5, 10, 20, 40, 80):
        yy = int(y1 - (np.sqrt(m / D_MAX)) * (y1 - y0))
        cv2.line(img, (8, yy), (32, yy), (255, 255, 255), 1)
        cv2.putText(img, f"{m} m", (36, yy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "depth", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def depth_video(run: Path, out: Path, fps: float) -> None:
    proc = None
    for t, bgr, d, inst in frames(run):
        col = colorize(d)
        edge = cv2.morphologyEx((inst > 0).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        col[edge & (inst != 0)] = (255, 255, 255)
        cv2.putText(bgr, f"t = {t:5.2f} s", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(col, "MoGe-2 metric depth, fixed 0-80 m (sqrt)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        img = np.hstack([bgr, col, legend(bgr.shape[0])])
        if proc is None:
            proc = writer(out, (img.shape[1], img.shape[0]), fps)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    proc.wait()


BEV_W, BEV_H, PX_M, BACK = 420, 480, 7.0, 8.0


def to_px(rel: np.ndarray) -> tuple[int, int]:
    return int(BEV_W / 2 - rel[1] * PX_M), int(BEV_H - (rel[0] + BACK) * PX_M)


def bev_frame(k: int, t: float, ego: np.ndarray, tracks: list[dict], path: tuple) -> np.ndarray:
    img = np.full((BEV_H, BEV_W, 3), 28, np.uint8)
    c, s = np.cos(ego[k, 2]), np.sin(ego[k, 2])
    R = np.array([[c, s], [-s, c]])

    def rel(p):
        return (np.atleast_2d(p) - ego[k, :2]) @ R.T

    for m in range(0, 70, 10):
        yy = to_px(np.array([m, 0]))[1]
        cv2.line(img, (0, yy), (BEV_W, yy), (50, 50, 50), 1)
        cv2.putText(img, f"{m}", (3, yy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1, cv2.LINE_AA)
    pp = rel(path)
    pp = pp[(pp[:, 0] > -BACK - 5) & (pp[:, 0] < 75)]
    for lane in (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5):
        side = pp + np.array([0, lane * 3.5 * 1.0])      # lane edges follow the ego path laterally
        pts = np.array([to_px(q) for q in side], np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, (70, 70, 70), 1, cv2.LINE_AA)
    if len(pp) > 1:
        cv2.polylines(img, [np.array([to_px(q) for q in pp], np.int32)], False, (0, 220, 255), 2, cv2.LINE_AA)
    for tr in tracks:
        if not tr["t0"] <= t <= tr["t1"] + 0.05:
            continue
        i = int(np.argmin(np.abs(tr["t"] - t)))
        col = COLORS.get(tr["category"], (200, 200, 200))
        j0 = max(0, i - 30)
        trail = rel(np.column_stack([tr["x"][j0:i + 1], tr["y"][j0:i + 1]]))
        if len(trail) > 1:
            cv2.polylines(img, [np.array([to_px(q) for q in trail], np.int32)], False, col, 1, cv2.LINE_AA)
        x, y = to_px(rel([tr["x"][i], tr["y"][i]])[0])
        cv2.circle(img, (x, y), 5, col, -1 if tr["observed"][i] else 1, cv2.LINE_AA)
        cv2.putText(img, f"{tr['id']} {tr['category'][:3]} {tr['speed'][i] * 3.6:.0f}", (x + 7, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
    ex, ey = to_px(np.array([0, 0]))
    cv2.fillPoly(img, [np.array([[ex, ey - 9], [ex - 6, ey + 6], [ex + 6, ey + 6]], np.int32)], (255, 255, 255))
    cv2.putText(img, f"ego {ego[k, 3] * 3.6:.0f} km/h  t={t:.1f}s", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def bev_video(run: Path, out: Path, ego: np.ndarray, ego_t: np.ndarray, tracks: list[dict], fps: float) -> None:
    proc = None
    for t, bgr, _, _ in frames(run):
        k = int(np.argmin(np.abs(ego_t - t)))
        img = np.hstack([bgr, bev_frame(k, t, ego, tracks, (ego[:, :2]))])
        if proc is None:
            proc = writer(out, (img.shape[1], img.shape[0]), fps)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    proc.wait()


def bev_png(out: Path, ego: np.ndarray, ego_t: np.ndarray, tracks: list[dict], title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(ego[:, 0], ego[:, 1], "k-", lw=2.5, label="ego (ViPE, despiked)")
    ax.scatter(ego[::15, 0], ego[::15, 1], c=ego_t[::15], cmap="viridis", s=18, zorder=3)
    cmap = {"car": "tab:blue", "truck": "tab:orange", "bus": "tab:purple", "motorcycle": "tab:olive",
            "bicycle": "tab:green", "person": "tab:red"}
    for tr in tracks:
        ax.plot(tr["x"], tr["y"], "-", color=cmap.get(tr["category"], "gray"), lw=1.5,
                alpha=0.5 + 0.5 * tr["confidence"])
        ax.annotate(f"{tr['id']}", (tr["x"][0], tr["y"][0]), fontsize=7, color=cmap.get(tr["category"], "gray"))
        ax.plot(tr["x"][-1], tr["y"][-1], ">", color=cmap.get(tr["category"], "gray"), ms=4)
    for cat, col in cmap.items():
        if any(tr["category"] == cat for tr in tracks):
            ax.plot([], [], "-", color=col, label=cat)
    ax.set_aspect("equal")
    ax.set_xlabel("x forward at t0 (m)")
    ax.set_ylabel("y left (m)")
    ax.set_title(f"{title}: {len(tracks)} NPC tracks")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    z = np.load(a.run / "track.npz")
    ego = vipe.despike(vipe.ego_track(z["poses"], z["t"]), z["t"])
    drops: dict = {}
    tracks = npc.npc_tracks(a.run, ego, z["t"], drops=drops)
    npc.save_tracks(tracks, a.out / "npc_tracks.json")
    (a.out / "npc_summary.json").write_text(json.dumps({"tracks": npc.npc_summary(tracks), "drops": drops}, indent=1))
    fps = float(np.load(next(a.run.glob("w*/dense.npz")))["fps"])
    bev_png(a.out / "bev.png", ego, z["t"], tracks, a.title or a.out.name)
    if not a.no_video:
        depth_video(a.run, a.out / "depth.mp4", fps)
        bev_video(a.run, a.out / "bev.mp4", ego, z["t"], tracks, fps)
    print(json.dumps({"tracks": len(tracks), "drops": drops}))


if __name__ == "__main__":
    main()

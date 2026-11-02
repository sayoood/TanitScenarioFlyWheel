"""Queue of dense ViPE runs (thor3): wait for a quiet GPU, run ``run_vipe(dense=True)``, write ``summary.json``.

    python run_dense.py RUN_DIR=VIDEO[:MAX_SECONDS][@FOCAL_PX][%OPT,...] ...    # RUN_DIR/w00..., summary.json

OPT: ``nodense`` (poses only), ``height=360`` (processed clip height), or a worker flag with its value after ``=``,
e.g. ``--override=pipeline.slam.ba.robust_kernel=huber`` or ``--kf-gap-s=1.0``.

Before every window launch the job claims an exclusive Thor window (``~/tsf/dev/heavy_ok.sh vipe-agent``), and no
other ``tsf-vipe`` image container (the API's UC2 runs, the other ViPE agent) and no ``tsf-cosmos-*`` container may
be running; the 30 GB host-memory floor of ``run_vipe`` applies as well. The window is released after each job.
``TSF_CLAIM_MIN=N`` instead takes ONE window of N minutes for the whole queue, starts no job that would not fit
(``TSF_JOB_MIN``, default 9 min) and releases at the end.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from tsf.ingest import vipe


def busy() -> list[str]:
    rows = subprocess.run(["docker", "ps", "--format", "{{.Image}} {{.Names}}"], capture_output=True,
                          text=True, check=False).stdout.splitlines()
    return [r for r in rows if r.startswith("tsf-vipe") or ("cosmos" in r.lower() and not r.endswith("-dl"))]


HEAVY_OK = Path.home() / "tsf" / "dev" / "heavy_ok.sh"
WINDOW_MIN = float(os.environ.get("TSF_CLAIM_MIN", "0"))
JOB_MIN = float(os.environ.get("TSF_JOB_MIN", "9"))


def heavy(*args: str) -> bool:
    """``heavy_ok.sh vipe-agent [--claim N | --release]`` (Thor scheduling between agents); True without the script."""
    return not HEAVY_OK.exists() or subprocess.run([str(HEAVY_OK), "vipe-agent", *args], check=False).returncode == 0


def gate(poll_s: float = 15.0) -> None:
    """Claim an exclusive window (the other ViPE agent queues behind this job), then wait for running containers."""
    t0 = time.time()
    while not WINDOW_MIN and not heavy("--claim", "25"):
        time.sleep(poll_s)
    while (b := busy()):
        print(f"[{time.strftime('%H:%M:%S')}] waiting for {b} ({time.time() - t0:.0f} s)", flush=True)
        time.sleep(poll_s)


def release(last: bool, pause_s: float = 75.0) -> None:
    """Free the window; between jobs leave the other ViPE agent time to start its next container."""
    if WINDOW_MIN:
        return
    heavy("--release")
    if not last:
        time.sleep(pause_s)


def main() -> None:
    deadline = None
    if WINDOW_MIN:
        while not heavy("--claim", f"{WINDOW_MIN:g}"):
            time.sleep(15)
        deadline = time.time() + WINDOW_MIN * 60
        print(f"[{time.strftime('%H:%M:%S')}] claimed {WINDOW_MIN:g} min", flush=True)
    try:
        run_queue(sys.argv[1:], deadline)
    finally:
        if WINDOW_MIN:
            heavy("--release")
            print(f"[{time.strftime('%H:%M:%S')}] window released", flush=True)


def run_queue(specs: list[str], deadline: float | None) -> None:
    for k, spec in enumerate(specs):
        if deadline and deadline - time.time() < JOB_MIN * 60:
            left = (deadline - time.time()) / 60
            print(f"[{time.strftime('%H:%M:%S')}] {left:.0f} min left: not starting {spec[:90]}", flush=True)
            continue
        run_dir, video = spec.split("=", 1)
        video, _, flags = video.partition("%")
        video, _, focal = video.partition("@")
        video, _, secs = video.partition(":")
        out = Path(run_dir)
        opts = [x for x in flags.split(",") if x]
        dense = "nodense" not in opts
        height = next((int(x.split("=")[1]) for x in opts if x.startswith("height=")), 480)
        wargs = tuple(y for x in opts if x.startswith("--") for y in x.partition("=")[::2] if y)
        print(f"[{time.strftime('%H:%M:%S')}] {out.parent.name}/{out.name}: {video} focal={focal or 'estimated'}",
              flush=True)
        try:
            res = vipe.run_vipe(video, out, max_seconds=float(secs) if secs else None, fps=15, dense=dense,
                                height=height, gate=gate, retries=2, focal_px=float(focal) if focal else None,
                                worker_args=wargs)
        except RuntimeError as e:
            print(f"[{time.strftime('%H:%M:%S')}] {out.parent.name}/{out.name} FAILED: {str(e)[:300]}", flush=True)
            release(k == len(specs) - 1)
            continue
        keys = ("start_s", "override", "kf_gap_s", "frames", "pipeline_s", "dense_s", "dense_fps", "infer_fps",
                "gpu_peak_allocated_gb", "gpu_peak_reserved_gb", "rss_peak_gb", "instances", "model_load_s")
        summary = {"video": video, "frames": res["frames"], "runtime_s": res["runtime_s"],
                   "camera_height": res["camera_height"], "intrinsics": res["intrinsics"],
                   "windows": [{k: w["stats"].get(k) for k in keys} | {"runtime_s": w["runtime_s"]}
                               for w in res["windows"]]}
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary), flush=True)
        release(k == len(specs) - 1)


if __name__ == "__main__":
    main()

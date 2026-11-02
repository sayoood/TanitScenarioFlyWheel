"""Camera + LiDAR (3 views, 97 frames, chained 49-frame windows) for the TSIR of finished API runs, registered in run.json.

Same registration as the ``sensors`` stage of ``tsf.server.runs`` (stage ``sensors`` = ok, artifacts ``mosaic.mp4`` /
``lidar_bev.mp4`` / ``sensors.json`` in the run directory, ``run["sensors"]``, kpis), but for runs that were made without
``options.sensors``. thor3 only; the Cosmos slot and the unified memory are shared, so it is two phases:

    prepare RUN...   CPU: TanitSim -> RDS-HQ -> WSM -> LiDAR control -> BEV, plus LLM captions (needs >= 35 GB available)
    render  RUN...   needs a Sim-Auto server sized for joint requests (``pipeline.ensure_server(views=3, lidar=True)``)
    status  RUN...   what exists

``RUN`` is a run id or directory below ``runs/api``. The positive TSIR is ``tsir.json`` (outputs in ``sensors/``); a
``negative_tsir.json`` (UC3 / UC5 twins) is rendered into ``sensors_negative/`` and registered as ``run["sensors_negative"]``.
Outputs of runs that come from recorded data (UC2 / UC4 / UC5) stay on the machine they were rendered on.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path

RUNS = Path(os.environ.get("TSF_RUNS", Path(__file__).resolve().parents[1] / "runs" / "api"))
FRAMES = 97


def run_dir(r: str) -> Path:
    p = Path(r)
    return p if p.is_dir() else RUNS / r


def targets(rd: Path, which: str = "both") -> list[tuple[str, Path, Path]]:
    """``[(key, tsir path, out dir)]``: ``sensors`` for ``tsir.json``, ``sensors_negative`` for ``negative_tsir.json``."""
    t = []
    if which in ("both", "positive") and (rd / "tsir.json").exists():
        t.append(("sensors", rd / "tsir.json", rd / "sensors"))
    if which in ("both", "negative") and (rd / "negative_tsir.json").exists():
        t.append(("sensors_negative", rd / "negative_tsir.json", rd / "sensors_negative"))
    return t


def _child_rss_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20, 2)


def prepare(rd: Path, which: str, cams, frames: int) -> list[dict]:
    from tsf.cosmos import pipeline as P
    from tsf.cosmos.captions import write_sidecars
    out = []
    for key, tsir_path, odir in targets(rd, which):
        t = time.perf_counter()
        tsir, txt = P._as_tsir(tsir_path)
        info, _, _ = P.prepare_geometry(txt, odir, cams, frames, clip_id=tsir["meta"]["id"])
        rep = {"run": rd.name, "target": key, "geometry": info["timings"], "geometry_reused": info["reused"],
               "peak_main_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
               "peak_child_rss_gb": _child_rss_gb()}
        cap_file = odir / "captions_source.json"
        if not cap_file.exists():
            t1 = time.perf_counter()
            caps, src = P.make_captions(tsir, list(cams), frames, None)
            if src == "given" or not isinstance(caps, dict):
                raise RuntimeError("unexpected caption result")
            write_sidecars(caps, odir / "captions", frames=frames)
            cap_file.write_text(json.dumps({"source": src, "seconds": round(time.perf_counter() - t1, 1)}))
        rep["captions"] = json.loads(cap_file.read_text())
        rep["seconds"] = round(time.perf_counter() - t, 1)
        print(json.dumps(rep), flush=True)
        out.append(rep)
    return out


OWNER = os.environ.get("TSF_OWNER", "lidar-sensors-agent")


def server_up(cams, wait_s: float = 1800.0) -> dict:
    """Our Sim-Auto server is running (started here if needed, never someone else's): returns ``ensure_server``'s dict.
    Waits for the memory a start needs; another owner's server is waited for, not used."""
    from tsf.cosmos import generate as g
    from tsf.cosmos import pipeline as P
    t = time.perf_counter()
    while True:
        running = g.running_servers()
        if running and not (running == ["tsf-cosmos-sim"] and g.server_owner() == OWNER):
            why = f"a Cosmos server of another owner is running: {running} ({g.server_owner(running[0])!r})"
        else:
            try:
                return P.ensure_server(views=len(cams), lidar=True, owner=OWNER)
            except P.CosmosBusyError as exc:
                why = str(exc)[:160]
        if time.perf_counter() - t > wait_s:
            raise P.CosmosBusyError(why)
        print(f"[server] {why}; retry in 30 s", flush=True)
        time.sleep(30)


def render(rd: Path, which: str, cams, frames: int, seed: int = 0, attempts: int = 3) -> list[dict]:
    """Render every target of the run. The Cosmos server may be killed by the memory guard in the middle of a window
    (``mem_guard2`` at 6 GB available): the server is brought back and the run repeated; windows that finished are reused."""
    import httpx

    from tsf.cosmos import pipeline as P
    out = []
    for key, tsir_path, odir in targets(rd, which):
        cap_file = odir / "captions_source.json"
        caps = odir / "captions" if cap_file.exists() else None
        t = time.perf_counter()
        for attempt in range(1, attempts + 1):
            try:
                server_up(cams)
                res = P.render_sensors(tsir_path, odir, cameras=tuple(cams), frames=frames, captions=caps, seed=seed,
                                       geometry_dir=odir)
                break
            except (httpx.TransportError, httpx.HTTPStatusError, RuntimeError) as exc:
                print(f"[render] {rd.name}/{key} attempt {attempt}/{attempts} failed: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                if attempt == attempts:
                    raise
                time.sleep(15)
        seconds = round(time.perf_counter() - t, 1)
        if cap_file.exists():                       # the sidecars came from the LLM / template captioner, not from the caller
            src = json.loads(cap_file.read_text())["source"]
            res["captions"]["source"] = src
            sj = json.loads((odir / "sensors.json").read_text())
            sj["captions"]["source"] = src
            (odir / "sensors.json").write_text(json.dumps(sj, indent=1, default=str))
        reg = register(rd, key, res, seconds)
        print(json.dumps(reg), flush=True)
        out.append(reg)
    return out


def _lidar_numbers(res: dict) -> dict:
    cs = (res.get("lidar") or {}).get("checks_summary") or {}
    s, o = cs.get("summary") or {}, cs.get("objects") or {}
    keep = ("sweeps", "mean_points_per_sweep", "mean_valid_fraction", "median_abs_range_err_m", "p90_abs_range_err_m",
            "control_agree_within_1m", "ground_present_sweeps", "ground_height_m_median")
    return {**{k: s[k] for k in keep if k in s}, **{k: o[k] for k in ("pairs", "recall_ge10", "median_points_generated") if k in o},
            "warnings": cs.get("warnings", [])}


def register(rd: Path, key: str, res: dict, seconds: float) -> dict:
    """Copy mosaic / LiDAR BEV / sensors.json into the run directory and record them in run.json (what the ``sensors``
    stage of ``tsf.server.runs`` does)."""
    from tsf.server.runs import artifacts, now
    positive = key == "sensors"
    pre = "" if positive else "negative_"
    lid = res.get("lidar") or {}
    for src, dst in ((res.get("mosaic"), "mosaic.mp4"), (lid.get("bev_video"), "lidar_bev.mp4"),
                     (rd / key / "sensors.json", "sensors.json")):
        if src and Path(src).exists():
            shutil.copy2(src, rd / (pre + dst))
    warn = (lid.get("checks_summary") or {}).get("warnings", [])
    frames = res["frames"]
    rec = {"cameras": res["cameras"], "frames": frames, "t0_s": res["t0_s"], "timings": res["timings"],
           "lidar_warnings": warn, "errors": res.get("errors") or {}, "captions": res["captions"]["source"],
           "windows": [{k: w[k] for k in ("start_frame", "frames", "request_s", "mem_used_peak_delta_gb")} for w in res.get("windows") or []],
           "lidar": _lidar_numbers(res), "weather_physics": (lid.get("weather") or {}).get("physics_applied"),
           "dir": key}
    p = rd / "run.json"
    run = json.loads(p.read_text())
    run[key] = rec
    if positive:
        run.setdefault("kpis", {}).update(cosmos_s_per_frame=round(res["timings"]["cosmos_s"] / frames, 2), lidar_checks_ok=not warn)
        st = next((s for s in run["stages"] if s["name"] == "sensors"), None)
        if st is None:
            st = {"name": "sensors"}
            run["stages"].append(st)
        st.update(status="ok", started=now(), seconds=seconds,
                  detail=f"{len(res['cameras'])} cameras{' + LiDAR' if lid.get('enabled') else ''}, "
                         f"{res['timings']['cosmos_s']:.0f} s Cosmos" + (f"; {len(warn)} LiDAR warning(s)" if warn else ""))
    run["artifacts"] = artifacts(rd)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(run, indent=1, default=str))
    os.replace(tmp, p)
    return {"run": rd.name, "target": key, "seconds": seconds, "timings": res["timings"], "lidar": rec["lidar"],
            "windows": rec["windows"], "errors": rec["errors"], "captions": rec["captions"]}


def status(rd: Path, which: str) -> dict:
    d = json.loads((rd / "run.json").read_text())
    st = next((s for s in d["stages"] if s["name"] == "sensors"), {})
    return {"run": rd.name, "uc": d["uc"], "sensors_stage": st.get("status"), "targets": {
        key: {"geometry": (odir / "geometry.json").exists(), "bev": (odir / "bev.mp4").exists(),
              "captions": (odir / "captions_source.json").exists(), "rendered": (odir / "sensors.json").exists(),
              "registered": key in d} for key, _, odir in targets(rd, which)}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["prepare", "render", "status"])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--which", choices=["both", "positive", "negative"], default="both")
    ap.add_argument("--frames", type=int, default=FRAMES)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from tsf.cosmos import generate as g
    from tsf.cosmos import pipeline as P
    cams = [c for c in g.CAMERA_ORDER if c in P.SENSOR_CAMERAS]        # the order render_sensors uses (geometry key)
    for r in a.runs:
        rd = run_dir(r)
        if a.cmd == "prepare":
            prepare(rd, a.which, cams, a.frames)
        elif a.cmd == "render":
            render(rd, a.which, cams, a.frames, a.seed)
        else:
            print(json.dumps(status(rd, a.which)))


if __name__ == "__main__":
    sys.exit(main())

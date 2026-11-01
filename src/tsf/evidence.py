"""Visual evidence for the flywheel's positive results: labelled side-by-side videos rendered by TanitSim.

    python -m tsf.evidence uc5 <run_id>         lifted NEGATIVE log | POSITIVE twin (same NPCs, new ego plan)
    python -m tsf.evidence uc3 <run_id>         verified positive twin of an incident (+ closed loop)
    python -m tsf.evidence uc2 <run_id>         dash-cam clip | re-created scenario (ViPE ego + VLM scene)
    python -m tsf.evidence closed_loop <run_id> closed-loop policies side by side
    python -m tsf.evidence ablation             one-shot (A) | agentic + repair (D/E), where the agent wins
    python -m tsf.evidence all                  every eligible run and ablation query -> <out>/index.json

The title bar of each panel carries the verdict (valid / failure, key metric). Videos derived from
PhysicalAI-AV, IMDAD or YouTube clips stay on thor3 and are served by the API (``public: false`` in the index);
text-query and example results are synthetic and may be published.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

from tsf.verify import ladder

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs" / "api"
OUT = REPO / "runs" / "evidence"
ABLATION = Path.home() / "tsf/dev/ablation"
SIZE = (960, 544)                                   # multiple of 16 for the encoder


def _ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()          # libx264 build (thor's apt ffmpeg is NVENC-only)


def render(tsir: dict, out: Path, title: str, wd: Path | None = None, fps: int = 20) -> Path:
    """Simulate a TSIR and render the BEV; a scenario that cannot be simulated becomes a still card with the
    verifier's first failure, so a failed baseline is still visible next to the agent's result."""
    from tsf.sim.render2d import render_video
    rep = ladder.run(json.loads(json.dumps(tsir)), wd or out.with_suffix(""), qc=False)
    if hasattr(rep, "_log"):
        return render_video(rep._sim, rep._log, out, fps=fps, title=title[:80], size=SIZE)
    return card(out, title, "NOT SIMULATABLE", rep.diagnostics()[:400])


def card(out: Path, title: str, head: str, body: str, seconds: float = 6.0, fps: int = 20) -> Path:
    import imageio.v2 as iio2
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", SIZE, (18, 22, 28))
    d = ImageDraw.Draw(img)
    try:
        from tsf.sim.render2d import FONT_DIR
        f_big, f = (ImageFont.truetype(str(FONT_DIR / "NotoSans-Bold.ttf"), s) for s in (34, 18))
    except Exception:  # noqa: BLE001 - fall back to the PIL default font
        f_big = f = ImageFont.load_default()
    d.text((32, 24), title[:80], font=f, fill=(170, 180, 190))
    d.text((32, 200), head, font=f_big, fill=(235, 90, 70))
    for k, line in enumerate(textwrap.wrap(body, 80)[:8]):
        d.text((32, 260 + 26 * k), line, font=f, fill=(220, 220, 220))
    frame = np.asarray(img)
    out.parent.mkdir(parents=True, exist_ok=True)
    with iio2.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=8) as w:
        for _ in range(int(seconds * fps)):
            w.append_data(frame)
    return out


def stack(videos: list[Path], out: Path, cols: int | None = None) -> Path:
    """Tile videos (scaled to one panel size) left-to-right / in a grid; shorter ones hold their last frame."""
    n = len(videos)
    cols = cols or min(n, 2)
    w, h = SIZE
    flt = "".join(f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,"
                  f"setsar=1,fps=20[v{i}];" for i in range(n))
    if n == 1:
        flt += "[v0]copy[out]"
    else:
        lay = "|".join(f"{(i % cols) * w}_{(i // cols) * h}" for i in range(n))
        flt += "".join(f"[v{i}]" for i in range(n)) + f"xstack=inputs={n}:layout={lay}:fill=black[out]"
    cmd = [_ffmpeg(), "-loglevel", "error", "-y"] + sum((["-i", str(v)] for v in videos), []) + [
        "-filter_complex", flt, "-map", "[out]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out)]
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    return out


def _run(rid: str) -> tuple[dict, Path]:
    p = RUNS / rid
    return json.loads((p / "run.json").read_text()), p


def uc5(rid: str) -> dict:
    run, p = _run(rid)
    k = run.get("kpis") or {}
    neg = json.loads((p / "negative_tsir.json").read_text())
    pos = json.loads((p / "tsir.json").read_text())
    wd = OUT / rid
    a = render(neg, wd / "negative.mp4", f"NEGATIVE (lifted log): peak decel {k.get('neg_peak_decel', 0):.1f} m/s2")
    b = render(pos, wd / "positive.mp4", f"POSITIVE twin: peak decel {k.get('pos_peak_decel', 0):.1f} m/s2, "
                                         f"IMDAD package {'valid' if run.get('package_valid') else 'invalid'}")
    return {"kind": "uc5", "run": rid, "video": str(stack([a, b], wd / "uc5_negative_vs_positive.mp4")),
            "public": False, "source": run["input"].get("snippet")}


def uc3(rid: str) -> dict:
    run, p = _run(rid)
    vids = [p / "bev.mp4"] + sorted(p.glob("closed_loop_*.mp4"))
    vids = [v for v in vids if v.exists()]
    return {"kind": "uc3", "run": rid, "video": str(stack(vids, OUT / rid / "uc3_twin.mp4")), "public": False,
            "source": run["input"].get("incident_id")}


def uc2(rid: str) -> dict:
    from tsf.agents.video import resolve
    run, p = _run(rid)
    src = resolve(run["input"]["video"])
    wd = OUT / rid
    wd.mkdir(parents=True, exist_ok=True)
    dur = (run.get("metrics") or {}).get("duration") or 20
    clip = wd / "dashcam.mp4"
    subprocess.run([_ffmpeg(), "-loglevel", "error", "-y", "-i", str(src), "-t", str(dur), "-an", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(clip)], check=True)
    return {"kind": "uc2", "run": rid, "video": str(stack([clip, p / "bev.mp4"], wd / "uc2_dashcam_vs_scenario.mp4")),
            "public": False, "source": run["input"]["video"]}


def closed_loop(rid: str) -> dict | None:
    run, p = _run(rid)
    vids = sorted(p.glob("closed_loop_*.mp4"))
    if not vids:
        return None
    return {"kind": "closed_loop", "run": rid, "video": str(stack(vids, OUT / rid / "closed_loop.mp4")),
            "public": run["uc"] == "uc1", "source": run["input"].get("query") or run["input"].get("example")}


def ablation() -> list[dict]:
    """One panel per condition for every (query, repetition) where the agent is correct and one-shot is not."""
    from tsf.agents.ablation import QUERIES
    rows = [json.loads(ln) for ln in (ABLATION / "rows.jsonl").read_text().splitlines() if ln.strip()]
    by = {(r["cond"], r["q"], r["rep"]): r for r in rows}
    out = []
    for (cond, q, rep), r in sorted(by.items()):
        if cond not in ("D", "E") or not r["correct"]:
            continue
        base = by.get(("A", q, rep))
        if base is None or base["correct"]:
            continue
        panels = []
        for c, row in (("A", base), (cond, r)):
            wd = ABLATION / "work" / f"{c}_q{q}_r{rep}"
            att = sorted(wd.glob("attempt*/tsir.json"))
            name = {"A": "one-shot", "D": "agentic + repair", "E": "agentic + repair, Harness2"}[c]
            verdict = "correct" if row["correct"] else (row["intent_failed"] or [row["first_failure"]])[0]
            title = f"{c} {name}: {verdict}"[:80]
            dst = OUT / "ablation" / f"q{q}_r{rep}_{c}.mp4"
            if not att:
                panels.append(card(dst, title, "TSIR NOT RECORDED", row["first_failure"]))
                continue
            final = att[-1] if c != "A" else att[0]
            panels.append(render(json.loads(final.read_text()), dst, title, wd=dst.with_suffix("")))
        out.append({"kind": "ablation", "query": QUERIES[q][0], "q": q, "rep": rep, "split": QUERIES[q][1],
                    "video": str(stack(panels, OUT / "ablation" / f"q{q}_r{rep}_A_vs_{cond}.mp4")), "public": True})
    return out


def main(argv: list[str]) -> None:
    kind, args = argv[0], argv[1:]
    if kind == "all":
        idx = []
        for p in sorted(RUNS.glob("*/run.json")):
            run = json.loads(p.read_text())
            if run["status"] != "ok":
                continue
            fn = {"uc5": uc5, "uc3": uc3, "uc2": uc2}.get(run["uc"])
            for f in ([fn] if fn else []) + [closed_loop]:
                try:
                    r = f(run["id"])
                    if r:
                        idx.append(r)
                        print(r["kind"], r["video"], flush=True)
                except Exception as e:  # noqa: BLE001 - one bad run must not stop the index
                    print("skip", run["id"], f.__name__, type(e).__name__, str(e)[:160], flush=True)
        if (ABLATION / "rows.jsonl").exists():
            idx += ablation()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "index.json").write_text(json.dumps(idx, indent=1))
        print(f"{len(idx)} evidence videos -> {OUT / 'index.json'}")
    elif kind == "ablation":
        print(json.dumps(ablation(), indent=1))
    else:
        print(json.dumps({"uc5": uc5, "uc3": uc3, "uc2": uc2, "closed_loop": closed_loop}[kind](args[0]), indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])

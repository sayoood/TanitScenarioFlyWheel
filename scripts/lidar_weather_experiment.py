"""Does a weather caption change the LiDAR that Cosmos3-Nano-Sim-Auto generates jointly with the cameras? (thor3 only)

Same HD-map LiDAR control, same WSM controls, same seed, only the caption differs (``pipeline.environment_captions``:
clear / rain / heavy rain / fog 60 m / snow, all at 11:30 so that only the weather changes), plus a second clear run
with another seed as the noise floor. Metrics per variant (``lidar.cloud_metrics``): points per sweep, max / p99 range,
share of returns beyond 50 m, mean intensity and near-range clutter (airborne returns < 15 m that the control does not
explain). ``lidar_weather.compare_variants`` turns them into deltas, ratios and ``effect_over_noise``.

Needs a Sim-Auto server sized for joint requests (``pipeline.ensure_server(views=3, lidar=True)``) and the geometry of
``pipeline.prepare_geometry`` (``--geometry``). Output: ``<out>/<variant>/`` (cosmos run) and ``<out>/weather_effect.json``.

    python scripts/lidar_weather_experiment.py --geometry ~/tsf/dev/sensors/uc1_geometry --out ~/tsf/dev/sensors/weather_exp
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

VARIANTS = {                                     # name -> (appearance override, seed)
    "clear": ({"weather": "clear", "time_of_day": "11:30"}, 0),
    "clear_s1": ({"weather": "clear", "time_of_day": "11:30"}, 1),
    "rain": ({"weather": "rain", "time_of_day": "11:30", "road_surface": "wet"}, 0),
    "heavy_rain": ({"weather": "heavy_rain", "time_of_day": "11:30", "road_surface": "wet"}, 0),
    "fog": ({"weather": "fog", "fog_visibility_m": 60.0, "time_of_day": "11:30"}, 0),
    "snow": ({"weather": "snow", "time_of_day": "11:30", "road_surface": "snow"}, 0),
}


def _camera_stats(video) -> dict:
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
    ok, fr = cap.read()
    cap.release()
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    return {"mean_brightness": round(float(np.asarray(fr).mean()), 1), "mean_saturation": round(float(hsv[..., 1].mean()), 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--geometry", required=True, help="pipeline.prepare_geometry dir (rds_hq/, wsm/, lidar/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=49)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--url", default=None)
    a = ap.parse_args()

    from tsf.cosmos import generate as g
    from tsf.cosmos import lidar as L
    from tsf.cosmos import lidar_weather as W
    from tsf.cosmos import pipeline

    gdir, out = Path(a.geometry).expanduser(), Path(a.out).expanduser()
    geo = json.loads((gdir / "geometry.json").read_text())
    cid = geo["clip_id"]
    sid = json.loads((gdir / "rds_hq" / "tsf_export" / f"{cid}.json").read_text())["scenario"]
    cams = [c for c in g.CAMERA_ORDER if c in pipeline.SENSOR_CAMERAS]
    control = gdir / "lidar" / "controls" / cid / "lidar_control.safetensors"
    env0 = {"time_of_day": "11:30", "weather": "clear", "road_surface": "dry"}
    runs, cam_stats = {}, {}
    for name in a.variants.split(","):
        app, seed = VARIANTS[name]
        caps = pipeline.environment_captions(sid, cams, a.frames, {**env0, **app})
        r = g.sim_chunk(gdir / "wsm", out / name, cams, a.frames, "480", caps, seed=seed, scenario=sid, lidar_control=control,
                        **({"base_url": a.url} if a.url else {}))
        runs[name] = {"request_s": r["request_s"], "mem_used_peak_gb": r["mem_used_peak_gb"], "seed": seed, "appearance": app,
                      "server": r["lidar"]["server"]}
        cam_stats[name] = _camera_stats(r["videos"]["camera_front_wide_120fov"])
        print(name, runs[name], cam_stats[name], flush=True)
    sweeps = L.required_sweeps(a.frames)
    ctl = L.read_safetensors(out / next(iter(runs)) / "controls" / "lidar_control.safetensors")[0]["frames"]
    cmp_ = W.compare_variants({n: out / n / "lidar_generated.safetensors" for n in runs}, ctl, reference="clear",
                              noise_reference="clear_s1" if "clear_s1" in runs else None)
    res = {"frames": a.frames, "sweeps": sweeps, "runs": runs, "camera_front_last_frame": cam_stats, **cmp_}
    (out / "weather_effect.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in res if k not in ("runs",)}, indent=1))


if __name__ == "__main__":
    main()

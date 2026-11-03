"""Tests for the joint camera+LiDAR path: control contract, lattice, point clouds, weather post-process, request wire format.

Everything runs on synthetic data except ``test_real_control_*`` (skipped without the uc1 run under ``runs/wsm``).
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

from tsf.cosmos import lidar as L
from tsf.cosmos import lidar_weather as W

ROOT = Path(__file__).resolve().parents[1]
UC1 = ROOT / "runs/wsm/uc1_left_turn_intersection"


def _frames(t: int = 3, fill: float = 1.0) -> np.ndarray:
    """Synthetic [3, T, 128, 1800] map: a ground ring (range = 1.94 / -sin(el)) on the downward rays plus a wall at 20 m ahead."""
    d = L.ray_directions()
    f = np.zeros((3, t, L.ROWS, L.COLS), np.float32)
    down = d[..., 2] < -0.033                                             # ground ring out to ~59 m
    rg = np.where(down, L.LIDAR_Z_M / np.maximum(-d[..., 2], 1e-3), 0).astype(np.float32)
    rg[rg > 60] = 0
    for i in range(t):
        f[0, i] = rg
        f[0, i, 60:100, 880:920] = 20.0                                   # a patch straight ahead (columns ~ forward)
        f[1, i] = np.where(f[0, i] > 0, 0.4 * fill, 0)
        f[2, i] = (f[0, i] > 0).astype(np.float32)
    return f


def test_required_sweeps_matches_server_rule():
    assert [L.required_sweeps(n) for n in (97, 297, 49, 17, 33)] == [32, 99, 16, 6, 11]
    with pytest.raises(ValueError):
        L.required_sweeps(1)


def test_safetensors_roundtrip_and_header(tmp_path):
    a = np.random.default_rng(0).random((3, 2, 4, 5)).astype(np.float32)
    p = L.write_safetensors(tmp_path / "x.safetensors", {"frames": a}, {"k": "v"})
    got, meta = L.read_safetensors(p)
    assert meta == {"k": "v"} and np.array_equal(got["frames"], a)
    raw = p.read_bytes()
    n = int.from_bytes(raw[:8], "little")
    assert n % 8 == 0 and json.loads(raw[8:8 + n])["frames"]["dtype"] == "F32"
    st = pytest.importorskip("safetensors.numpy")                          # the real reader agrees (not in the repo venv)
    assert np.array_equal(st.load_file(str(p))["frames"], a)


def test_control_contract_validation_and_trim(tmp_path):
    f = _frames(5)
    ok = L.write_safetensors(tmp_path / "c.safetensors", {"frames": f})
    assert L.validate_control(ok) == ((3, 5, 128, 1800), 5)
    t = L.trim_control(ok, tmp_path / "t.safetensors", 3, start=1)
    got, _ = L.read_safetensors(t)
    assert got["frames"].shape == (3, 3, 128, 1800) and np.array_equal(got["frames"], f[:, 1:4])
    with pytest.raises(ValueError):
        L.trim_control(ok, tmp_path / "u.safetensors", 6)
    bad = L.write_safetensors(tmp_path / "b.safetensors", {"frames": f.astype(np.float64)})
    with pytest.raises(ValueError):
        L.validate_control(bad)
    bad2 = L.write_safetensors(tmp_path / "b2.safetensors", {"frames": f[:, :, :64]})
    with pytest.raises(ValueError):
        L.validate_control(bad2)
    f2 = f.copy()
    f2[1, 0, 0, 0] = 2.0
    with pytest.raises(ValueError):
        L.validate_control(L.write_safetensors(tmp_path / "b3.safetensors", {"frames": f2}))


def test_lattice_geometry():
    d = L.ray_directions()
    assert d.shape == (128, 1800, 3) and np.allclose(np.linalg.norm(d, axis=-1), 1, atol=1e-6)
    el = np.degrees(np.arcsin(d[..., 2]))
    assert el[0, 0] == pytest.approx(14.45, abs=0.05) and el[-1, 0] == pytest.approx(-24.96, abs=0.05)     # ring 0 = top
    az = np.degrees(np.arctan2(d[64, :, 1], d[64, :, 0]))
    assert abs(az[0] - 179.9) < 0.01                                      # column 0 looks backwards (leading edge + half cell)
    assert abs(az[899]) < 0.2 and abs(az[900]) < 0.2                      # centre columns look forward
    assert d[64, 450, 1] > 0 > d[64, 1350, 1]                             # left (y > 0) before forward, right after


def test_points_and_motion_compensation():
    f = _frames(1)
    c = L.sweep_points(f, 0)
    assert not bool(c["compensated"]) and len(c["xyz"]) == int((f[2, 0] > 0).sum())
    assert np.allclose(np.linalg.norm(c["xyz"], axis=1), c["range"], atol=1e-3)
    I = np.eye(4)
    T1 = np.eye(4)
    T1[0, 3] = 1.0                                                        # ego moves 1 m forward during the sweep
    same = L.sweep_points(f, 0, poses=(I, I))
    assert np.allclose(same["xyz"], c["xyz"], atol=1e-4) and bool(same["compensated"])
    moved = L.sweep_points(f, 0, poses=(I, T1))
    assert np.allclose(moved["xyz"] - c["xyz"], np.c_[c["t_rel"], 0 * c["t_rel"], 0 * c["t_rel"]], atol=1e-4)
    # a rotating sensor: yaw 10 deg over the sweep -> last-column points rotate by ~10 deg in the start frame
    yaw = np.radians(10)
    T2 = np.eye(4)
    T2[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    rot = L.sweep_points(f, 0, poses=(I, T2))
    last = c["col"] == c["col"].max()
    ang = np.degrees(np.arctan2(rot["xyz"][last, 1], rot["xyz"][last, 0]) - np.arctan2(c["xyz"][last, 1], c["xyz"][last, 0]))
    assert np.allclose((ang + 180) % 360 - 180, 10, atol=0.5)


def test_ground_fit_and_clutter_metric():
    f = _frames(2)
    cl = L.sweep_points(f, 0)
    g = L.fit_ground(cl["xyz"])
    assert L.ground_present(g) and g["height_m"] == pytest.approx(L.LIDAR_Z_M, abs=0.05) and g["tilt_deg"] < 1
    assert not L.ground_present(L.fit_ground(cl["xyz"][cl["xyz"][:, 2] > -0.5]))
    base = L.cloud_metrics(f, f)["mean"]
    assert base["near_clutter"] == 0
    noisy = f.copy()
    noisy[0, :, 20, 100:110] = 8.0                                         # airborne returns the control does not explain
    noisy[2, :, 20, 100:110] = 1.0
    assert L.cloud_metrics(noisy, f)["mean"]["near_clutter"] > 0


def test_object_box_membership():
    o0 = {"T": np.array([[1, 0, 0, 10.0], [0, 1, 0, 0], [0, 0, 1, 0.75], [0, 0, 0, 1]]), "lwh": np.array([4.0, 2.0, 1.5])}
    o1 = {"T": o0["T"] + np.array([[0, 0, 0, 3.0], [0] * 4, [0] * 4, [0] * 4]), "lwh": o0["lwh"]}
    pts = np.array([[10.0, 0, 0.5], [8.2, 0, 0.5], [13.0, 0, 0.5], [10, 3, 0.5], [10, 0, 3.0]])
    assert L.points_in_box(pts, np.zeros(5), o0, None).tolist() == [True, True, False, False, False]
    assert L.points_in_box(pts, np.ones(5), o0, o1).tolist() == [False, False, True, False, False]   # the box moved 3 m ahead


def test_bev_and_range_frames_render():
    f = _frames(1)
    cl = L.sweep_points(f, 0)
    img = L.render_bev(cl["xyz"], cl["intensity"], [(np.array([[12, 1], [12, -1], [8, -1], [8, 1.0]]), "Car")], title="t")
    assert img.shape == (800, 800, 3) and img.any()
    assert L.range_compare_frame(f, f, 0, "t").shape == (26 + 3 * 256, 1800, 3)


# ============================================================================== weather
def test_effective_environment_defaults():
    assert W.effective_environment({"weather": "heavy_rain"})["precipitation"] == 0.75
    assert W.effective_environment({"weather": "fog"})["fog_visibility_m"] == W.FOG_DEFAULT_VIS_M
    assert W.effective_environment({"weather": "fog", "fog_visibility_m": 50})["fog_visibility_m"] == 50
    assert not W.is_adverse({"weather": "clear"}) and not W.is_adverse({"weather": "cloudy", "cloud_cover": 1})
    assert W.is_adverse({"weather": "snow"}) and W.is_adverse({"weather": "fog", "fog_visibility_m": 80})


def test_degrade_clear_is_identity_and_input_untouched():
    f = _frames(2)
    f0 = f.copy()
    out, rep = W.degrade(f, {"weather": "clear"}, 0)
    assert np.array_equal(out, f) and "clear" in rep["note"]
    W.degrade(f, {"weather": "fog"}, 0)
    assert np.array_equal(f, f0)


@pytest.mark.parametrize("env", [{"weather": "fog", "fog_visibility_m": 60}, {"weather": "heavy_rain"}, {"weather": "snow"}])
def test_degrade_is_deterministic_and_valid(env):
    f = _frames(2)
    a, _ = W.degrade(f, env, 7)
    b, _ = W.degrade(f, env, 7)
    c, _ = W.degrade(f, env, 8)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert np.isfinite(a).all() and (a[0] >= 0).all() and (a[1:] >= 0).all() and (a[1:] <= 1).all()
    assert np.array_equal(a[2] > 0, a[0] > 0)                              # validity follows range
    assert a[0].max() <= L.MAX_RANGE_M


def test_fog_cuts_range_and_adds_backscatter_rain_drops_points():
    f = _frames(2)
    base = L.cloud_metrics(f, f)["mean"]
    fog, rep = W.degrade(f, {"weather": "fog", "fog_visibility_m": 60}, 0)
    m = L.cloud_metrics(fog, f)["mean"]
    assert m["max_range_m"] < 0.85 * base["max_range_m"] and m["frac_beyond_50m"] < 0.001 < base["frac_beyond_50m"]
    assert m["near_clutter"] > 0
    assert rep["totals"]["dropped"] > 0 and rep["totals"]["clutter_added"] > 0
    light, _ = W.degrade(f, {"weather": "rain", "precipitation": 0.1}, 0)
    heavy, _ = W.degrade(f, {"weather": "rain", "precipitation": 0.9}, 0)
    n_l, n_h = L.cloud_metrics(light, f)["mean"]["n_points"], L.cloud_metrics(heavy, f)["mean"]["n_points"]
    assert n_l > n_h and base["n_points"] > n_h                            # more rain, fewer far returns
    assert L.cloud_metrics(heavy, f)["mean"]["near_clutter"] > L.cloud_metrics(light, f)["mean"]["near_clutter"]


# ============================================================================== captions with environment
def test_environment_captions_night_rain_has_no_sunshine():
    from tsf.cosmos import pipeline
    caps = pipeline.environment_captions("uc1_left_turn_intersection", ["camera_front_wide_120fov", "camera_cross_left_120fov"], 97,
                                         {"weather": "heavy_rain", "time_of_day": "21:30", "road_surface": "dry"})
    for t in caps.values():
        low = t.lower()
        assert "rain" in low and "night" in low and "wet" in low
        assert all(w not in low for w in ("sunlit", "blue sky", "shadows", "sunlight")), t
        assert low.count("[") == 3 and "intersection" in low
    clear = pipeline.environment_captions("uc1_left_turn_intersection", ["camera_front_wide_120fov"], 97, {"weather": "clear"})
    assert "dry" in clear["camera_front_wide_120fov"].lower()


# ============================================================================== wire format
def test_joint_request_form():
    pytest.importorskip("httpx")
    from tsf.cosmos import generate as g
    ctl = {"camera_front_wide_120fov": Path("a.mp4"), "camera_cross_left_120fov": Path("b.mp4")}
    caps = {c: "x" for c in ctl}
    form, ups = g._build_request("sim", ctl, caps, frames=97, fps=30.0, res="480", seed=3, steps=4, guidance=1.0, shift=0.0,
                                 first_frames=None, lidar_control=Path("lidar_control.safetensors"))
    extra = json.loads(form["extra_params"])
    assert extra["lidar"] == {"control_reference_index": 2, "return_output": True}
    assert [u.name for u in ups] == ["a.mp4", "b.mp4", "lidar_control.safetensors"]
    assert form["fps"] == "30.0" and form["num_frames"] == "97" and extra["multiview"]["num_frames"] == 97
    form, ups = g._build_request("sim", ctl, caps, frames=97, fps=30.0, res="480", seed=3, steps=4, guidance=1.0, shift=0.0,
                                 first_frames={c: Path(c + ".png") for c in ctl}, lidar_control=Path("l.safetensors"),
                                 lidar_condition=Path("c.safetensors"))
    extra = json.loads(form["extra_params"])
    assert extra["lidar"] == {"control_reference_index": 4, "return_output": True, "condition_reference_index": 5}
    assert extra["multiview"]["views"][1]["vision_reference_index"] == 3
    with pytest.raises(ValueError):
        g._build_request("sim", ctl, caps, frames=97, fps=24.0, res="480", seed=0, steps=4, guidance=1.0, shift=0.0,
                         first_frames=None, lidar_control=Path("l.safetensors"))
    with pytest.raises(ValueError):
        g._build_request("transfer", ctl, caps, frames=97, fps=30.0, res="480", seed=0, steps=35, guidance=3.0, shift=10.0,
                         first_frames=None, lidar_control=Path("l.safetensors"))


class _FakeVideos(BaseHTTPRequestHandler):
    """Minimal /v1/videos async API: POST -> queued, first poll in_progress, then completed with a lidar descriptor."""
    polls = 0
    deleted = False
    fail = False

    def log_message(self, *a):
        pass

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self._send(200, json.dumps({"id": "video_gen_x", "status": "queued"}).encode())

    def do_GET(self):
        cls = type(self)
        if self.path == "/v1/videos/video_gen_x":
            cls.polls += 1
            if cls.fail:
                return self._send(500, json.dumps({"id": "video_gen_x", "status": "failed", "error": {"message": "boom"}}).encode())
            st = "in_progress" if cls.polls < 2 else "completed"
            lid = {"url": "/v1/videos/video_gen_x/lidar", "shape": [3, 2, 128, 1800]} if st == "completed" else None
            return self._send(200, json.dumps({"id": "video_gen_x", "status": st, "lidar": lid, "inference_time_s": 1.5}).encode())
        if self.path.endswith("/content"):
            return self._send(200, b"MP4DATA", "video/mp4")
        if self.path.endswith("/lidar"):
            return self._send(200, b"LIDARDATA", "application/octet-stream")
        self._send(404, b"{}")

    def do_DELETE(self):
        type(self).deleted = True
        self._send(200, b"{}")


@pytest.fixture
def fake_server():
    _FakeVideos.polls, _FakeVideos.deleted, _FakeVideos.fail = 0, False, False
    srv = HTTPServer(("127.0.0.1", 0), _FakeVideos)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_async_client_downloads_both_files_and_deletes_job(fake_server, tmp_path):
    pytest.importorskip("httpx")
    from tsf.cosmos import generate as g
    up = tmp_path / "lidar_control.safetensors"
    up.write_bytes(b"x")
    dt, job = g._post_async({"prompt": "p"}, [up], tmp_path / "cm.mp4", tmp_path / "l.safetensors", fake_server, 30, poll=0.01)
    assert (tmp_path / "cm.mp4").read_bytes() == b"MP4DATA" and (tmp_path / "l.safetensors").read_bytes() == b"LIDARDATA"
    assert job["status"] == "completed" and job["lidar"]["shape"] == [3, 2, 128, 1800] and _FakeVideos.deleted


def test_async_client_reports_failed_job(fake_server, tmp_path):
    pytest.importorskip("httpx")
    from tsf.cosmos import generate as g
    _FakeVideos.fail = True
    up = tmp_path / "a.safetensors"
    up.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="boom"):
        g._post_async({"prompt": "p"}, [up], tmp_path / "cm.mp4", tmp_path / "l.safetensors", fake_server, 30, poll=0.01)
    assert _FakeVideos.deleted


# ============================================================================== real control (Thor run)
needs_uc1 = pytest.mark.skipif(not (UC1 / "lidar/controls/uc1_left_turn_intersection/lidar_control.safetensors").exists(),
                               reason="uc1 LiDAR control run not available")


@needs_uc1
def test_real_control_meets_contract_and_lattice_roundtrip():
    ctl = UC1 / "lidar/controls/uc1_left_turn_intersection/lidar_control.safetensors"
    shape, t = L.validate_control(ctl)
    assert shape == (3, 99, 128, 1800) and t == 99
    f = L.read_safetensors(ctl)[0]["frames"]
    assert np.array_equal(f[1], f[2]) and set(np.unique(f[1, 0])) <= {0.0, 1.0}      # occupied placeholder channels
    sw = L.load_sweep_poses(UC1 / "lidar/lidar_rangemap_metadata/uc1_left_turn_intersection.npz")
    assert sw["frame_ids"][:3] == [0, 3, 6] and sw["poses"].shape == (99, 2, 4, 4)
    c = L.sweep_points(f, 5, poses=tuple(sw["poses"][5]))
    assert len(c["xyz"]) == int((f[0, 5] > 0).sum()) and np.abs(c["xyz"][:, 2]).max() < 30
    objs = L.read_objects(UC1 / "rds_hq", "uc1_left_turn_intersection", [0, 3])
    assert set(objs) == {0, 3} and "ego" not in objs[0]


# ============================================================================== server ownership / joint-server guard
def test_serve_labels_owner_and_sets_lidar_env():
    from tsf.cosmos import generate as g
    cmd = g.serve("sim", kv_views=3, lidar=True, dry_run=True, owner="unit-test")
    assert cmd[cmd.index("--label") + 1] == "tsf.owner=unit-test"
    assert "TSF_LIDAR_CHUNK=4" in cmd and "TSF_KV_LIDAR=1" in cmd and "TSF_KV_VIEWS=3" in cmd
    cam = g.serve("sim", kv_views=3, lidar=False, dry_run=True, owner="x")
    assert "TSF_LIDAR_CHUNK=0" in cam and "TSF_KV_LIDAR=0" in cam


def test_serve_refuses_to_replace_a_running_server(monkeypatch):
    from tsf.cosmos import generate as g
    started = []
    monkeypatch.setattr(g, "running_servers", lambda: ["tsf-cosmos-sim"])
    monkeypatch.setattr(g, "server_owner", lambda name="tsf-cosmos-sim": "someone-else")
    monkeypatch.setattr(g.subprocess, "run", lambda *a, **k: started.append(a))
    with pytest.raises(RuntimeError, match="someone-else"):
        g.serve("sim", kv_views=3, lidar=True)                        # same name, no force
    with pytest.raises(RuntimeError, match="another Cosmos server"):
        g.serve("transfer")                                             # different kind: always refused
    assert not started                                                  # nothing was removed or started


def test_serve_force_replaces_own_server(monkeypatch, tmp_path):
    from tsf.cosmos import generate as g
    calls = []
    monkeypatch.setattr(g, "running_servers", lambda: ["tsf-cosmos-sim"])
    monkeypatch.setattr(g.subprocess, "run", lambda cmd, **k: calls.append(cmd))
    g.serve("sim", kv_views=1, lidar=True, work_dir=tmp_path, owner="me", force=True)
    assert calls[0][:3] == ["docker", "rm", "-f"] and calls[1][:3] == ["docker", "run", "-d"]


def test_check_joint_server_requires_lidar_env(monkeypatch):
    from tsf.cosmos import generate as g
    monkeypatch.setattr(g, "server_owner", lambda name="tsf-cosmos-sim": "someone-else")
    good = {"TSF_KV_VIEWS": "3", "TSF_KV_LIDAR": "1", "TSF_LIDAR_CHUNK": "4"}
    monkeypatch.setattr(g, "server_env", lambda name="tsf-cosmos-sim": good)
    assert g.check_joint_server("http://127.0.0.1:8100")["TSF_LIDAR_CHUNK"] == "4"
    for bad in ({"TSF_KV_VIEWS": "3"}, {**good, "TSF_LIDAR_CHUNK": "0"}, {**good, "TSF_KV_LIDAR": "0"}):
        monkeypatch.setattr(g, "server_env", lambda name="tsf-cosmos-sim", bad=bad: bad)
        with pytest.raises(RuntimeError, match="joint camera"):
            g.check_joint_server("http://localhost:8100")
    monkeypatch.setattr(g, "server_env", lambda name="tsf-cosmos-sim": {})
    assert g.check_joint_server("http://127.0.0.1:8100") is None      # no such container: cannot be checked
    monkeypatch.setattr(g, "server_env", lambda name="tsf-cosmos-sim": {"TSF_KV_VIEWS": "3"})
    assert g.check_joint_server("http://other-host:8100") is None     # remote server: not ours to inspect


# ============================================================================== windows (clips longer than one request)
def test_window_caption_cuts_shifts_and_absorbs_slivers():
    from tsf.cosmos import generate as g
    cap = "[0s-1.1s]: A one.\n\n[1.1s-2.2s]: B two.\n\n[2.2s-3.23s]: C three."
    assert g.window_caption(cap, 0, 49 / 30) == "[0.00s-1.10s]: A one.\n\n[1.10s-1.63s]: B two."
    assert g.window_caption(cap, 48 / 30, 97 / 30) == "[0.00s-0.60s]: B two.\n\n[0.60s-1.63s]: C three."
    assert g.window_caption("[0s-1s]: A\n\n[1s-1.1s]: B\n\n[1.1s-3s]: C", 0, 3) == "[0.00s-1.00s]: A\n\n[1.00s-3.00s]: C"
    assert g.window_caption("no time blocks here", 1, 2) == "no time blocks here"


def test_sim_windows_chains_conditions_and_joins(tmp_path, monkeypatch):
    from tsf.cosmos import generate as g
    cam = "camera_front_wide_120fov"
    ctl = L.write_safetensors(tmp_path / "ctl.safetensors", {"frames": _frames(10)})
    CAP = {cam: "[0s-0.4s]: A\n\n[0.4s-0.83s]: B"}
    calls, guard = [], []

    def fake_chunk(wsm_dir, out_dir, cams, frames, res, captions, *, start, first_frames, seed, fps, scenario, lidar_control,
                   base_url, timeout):
        calls.append({"start": start, "frames": frames, "ff": first_frames, "seed": seed, "caps": captions[cam]})
        out = Path(out_dir)
        out.mkdir(parents=True)
        n = L.required_sweeps(frames)
        gp = L.write_safetensors(out / "lidar_generated.safetensors", {"frames": np.full((3, n, 128, 1800), float(len(calls)), np.float32)})
        (out / "v.mp4").write_bytes(b"x")
        (out / f"last_{cam}.png").write_bytes(b"x")
        res = {"request_s": 10.0, "videos": {cam: str(out / "v.mp4")}, "start_frame": start,
               "frames": frames, "mem_used_before_gb": 90.0, "mem_used_peak_gb": 100.0, "mem_used_peak_delta_gb": 10.0 + len(calls),
               "lidar": {"generated": str(gp), "sweeps": n, "descriptor": {"fps": 10.0}, "server": {"inference_time_s": 9.0,
                                                                                                    "peak_memory_mb": 1000.0 * len(calls)}}}
        (out / "result.json").write_text(json.dumps(res))
        return {**res, "last_frames": {cam: str(out / f"last_{cam}.png")}}
    monkeypatch.setattr(g, "sim_chunk", fake_chunk)
    monkeypatch.setattr(g, "concat_windows", lambda parts, dst, fps=30.0: Path(dst))
    monkeypatch.setattr(g, "probe", lambda p: {"frames": 25})
    monkeypatch.setattr(g, "find_controls", lambda d, cams: {c: Path(c) for c in cams})
    monkeypatch.setattr(g, "trim_control", lambda src, dst, **k: Path(dst))
    r = g.sim_windows(tmp_path, tmp_path / "out", [cam], 25, "480", CAP, window_frames=13, seed=5,
                      scenario="x", lidar_control=ctl, before_window=guard.append, drift_guard=False)
    assert guard == [0, 1] and [c["start"] for c in calls] == [0, 12] and [c["frames"] for c in calls] == [13, 13]
    assert calls[0]["ff"] is None and Path(calls[1]["ff"][cam]).name == f"last_{cam}.png" and [c["seed"] for c in calls] == [5, 6]
    assert calls[1]["caps"].startswith("[0.00s-") and "B" in calls[1]["caps"] and "A" not in calls[1]["caps"]
    gen = L.read_safetensors(r["lidar"]["generated"])[0]["frames"]
    assert gen.shape == (3, 8, 128, 1800)
    assert float(gen[:, :4].max()) == 1.0 and float(gen[:, 4:].min()) == 2.0
    assert L.read_safetensors(r["lidar"]["control"])[0]["frames"].shape[1] == 8
    assert r["request_s"] == 20.0 and r["mem_used_peak_delta_gb"] == 12.0 and len(r["windows"]) == 2
    assert r["lidar"]["server"]["peak_memory_mb"] == 2000.0 and r["lidar"]["descriptor"]["num_frames"] == 8
    # a repeated call (e.g. after a crash in a later window) reuses the finished windows and sends nothing
    def boom(*a, **k):
        raise AssertionError("window must be reused")
    monkeypatch.setattr(g, "sim_chunk", boom)
    again = g.sim_windows(tmp_path, tmp_path / "out", [cam], 25, "480", CAP, window_frames=13, seed=5,
                          scenario="x", lidar_control=ctl, before_window=guard.append, drift_guard=False)
    assert [w["resumed"] for w in again["windows"]] == [True, True] and guard == [0, 1]
    with pytest.raises(AssertionError):                                    # other captions: not reusable
        g.sim_windows(tmp_path, tmp_path / "out", [cam], 25, "480", {cam: "[0s-0.8s]: other"}, window_frames=13, seed=5,
                      scenario="x", lidar_control=ctl)
    with pytest.raises(ValueError):
        g.sim_windows(tmp_path, tmp_path / "bad", [cam], 25, "480", None, window_frames=14, scenario="x", lidar_control=ctl)


def test_coherent_appearance_adds_matching_road_surface():
    from tsf.cosmos import pipeline as P
    assert P._coherent_appearance(None) is None
    assert P._coherent_appearance({"weather": "heavy_rain", "time_of_day": "21:30"}) == {"weather": "heavy_rain", "time_of_day": "21:30",
                                                                                     "road_surface": "wet"}
    assert P._coherent_appearance({"weather": "snow"})["road_surface"] == "snow"
    assert P._coherent_appearance({"weather": "rain", "road_surface": "dry"})["road_surface"] == "dry"      # explicit wins
    assert P._coherent_appearance({"weather": "fog"}) == {"weather": "fog"}


# ============================================================================== run registration / BEV appearance
def test_bev_env_follows_the_rendered_appearance():
    from tsf.cosmos import pipeline as P
    tsir = {"environment": {"time_of_day": "11:30", "weather": "clear", "road_surface": "dry"}}
    base, eff, name = P._bev_env(tsir, None)
    assert base == eff == {"time_of_day": "11:30", "weather": "clear", "road_surface": "dry"} and name == "bev.mp4"
    assert P._bev_env(tsir, {"weather": "clear", "cloud_cover": 0.4})[2] == "bev.mp4"           # no visible change
    base, eff, name = P._bev_env(tsir, P._coherent_appearance({"weather": "heavy_rain", "time_of_day": "21:30"}))
    assert eff == {"time_of_day": "21:30", "weather": "heavy_rain", "road_surface": "wet"} and base["weather"] == "clear"
    assert name.startswith("bev_") and name.endswith(".mp4") and name != "bev.mp4"


def test_reused_geometry_makes_appearance_bev_without_a_long_wait(tmp_path, monkeypatch):
    """A new appearance on finished geometry only needs the BEV; it must not sit for ``wait_s`` (15 min) for 35 GB while a
    Cosmos server holds the memory - ``geometry_bev`` falls back to its low-memory render after a minute."""
    from tsf.cosmos import pipeline as P
    tsir = {"meta": {"id": "c1"}, "environment": {"time_of_day": "11:30", "weather": "clear", "road_surface": "dry"}}
    txt = json.dumps(tsir)
    cams = ["camera_front_wide_120fov"]
    (tmp_path / "wsm").mkdir()
    (tmp_path / "geometry.json").write_text(json.dumps({"key": P._geometry_key(txt, None, 97, cams, "hyperion", "resize_1280x720", True)}))
    seen = {}
    monkeypatch.setattr(P, "geometry_bev", lambda *a, **k: seen.update(k))
    info, sim, log = P.prepare_geometry(txt, tmp_path, cams, 97, appearance={"weather": "rain", "road_surface": "wet"})
    assert info["reused"] and sim is None and log is None
    assert seen["name"].startswith("bev_") and seen["env"]["weather"] == "rain" and seen["wait_s"] <= 60.0


def test_register_sensors_in_run_json(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("render_run_sensors", ROOT / "scripts" / "render_run_sensors.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rd = tmp_path / "run_x"
    (rd / "sensors").mkdir(parents=True)
    (rd / "sensors_negative").mkdir()
    (rd / "run.json").write_text(json.dumps({"id": "run_x", "uc": "uc3", "status": "ok", "kpis": {"epdms": 0.9},
                                              "stages": [{"name": "simulate", "status": "ok"}, {"name": "sensors", "status": "skipped",
                                                                                              "detail": "not requested (options.sensors)"}]}))
    for d in ("sensors", "sensors_negative"):
        (rd / d / "mosaic.mp4").write_bytes(b"m")
        (rd / d / "sensors.json").write_text("{}")
        (rd / d / "lidar").mkdir()
        (rd / d / "lidar" / "lidar_bev.mp4").write_bytes(b"b")

    def res(d):
        return {"cameras": ["a", "b", "c"], "frames": 97, "t0_s": 1.5, "timings": {"cosmos_s": 400.0}, "errors": {},
                "captions": {"source": "llm"}, "mosaic": str(rd / d / "mosaic.mp4"),
                "windows": [{"start_frame": 0, "frames": 49, "request_s": 200.0, "mem_used_peak_delta_gb": 9.0, "x": 1}],
                "lidar": {"enabled": True, "bev_video": str(rd / d / "lidar" / "lidar_bev.mp4"),
                          "checks_summary": {"summary": {"sweeps": 32, "mean_points_per_sweep": 1.0}, "objects": {"recall_ge10": 1.0},
                                             "warnings": []}, "weather": {"physics_applied": False}}}
    rec = mod.register(rd, "sensors", res("sensors"), 430.0)
    mod.register(rd, "sensors_negative", res("sensors_negative"), 410.0)
    run = json.loads((rd / "run.json").read_text())
    st = next(s for s in run["stages"] if s["name"] == "sensors")
    assert st["status"] == "ok" and st["seconds"] == 430.0 and "3 cameras + LiDAR" in st["detail"]
    assert run["sensors"]["lidar"]["sweeps"] == 32 and run["sensors_negative"]["dir"] == "sensors_negative"
    assert run["kpis"]["epdms"] == 0.9 and run["kpis"]["lidar_checks_ok"] is True and run["kpis"]["cosmos_s_per_frame"] == 4.12
    names = {a["name"] for a in run["artifacts"]}
    assert {"mosaic.mp4", "lidar_bev.mp4", "sensors.json", "negative_mosaic.mp4", "negative_lidar_bev.mp4"} <= names
    assert rec["lidar"]["recall_ge10"] == 1.0 and rec["windows"][0]["frames"] == 49


def test_effective_environment_fills_missing_fields():
    e = W.effective_environment({"weather": "rain"})
    assert e["time_of_day"] == "12:00" and e["road_surface"] == "dry" and e["precipitation"] > 0
    assert W.effective_environment(None)["weather"] == "clear"

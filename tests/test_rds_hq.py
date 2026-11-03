"""Tests for the TanitSim -> RDS-HQ exporter (``tsf.cosmos.rds_hq``).

The pure geometry tests need only numpy / shapely; tests that write a clip need the Cosmos framework checkout
(NVIDIA's ``rds_hq_io.py`` + Hyperion rig asset, ``TSF_WSM_DIR`` or the default path) and are skipped without it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tsf.cosmos import rds_hq

ROOT = Path(__file__).resolve().parents[1]


def _have_nvidia() -> bool:
    try:
        rds_hq.wsm_dir()
        return True
    except FileNotFoundError:
        return False


needs_nvidia = pytest.mark.skipif(not _have_nvidia(), reason="Cosmos WSM checkout not available")


@pytest.fixture(scope="module")
def uc1():
    from tsf.verify import ladder
    rep = ladder.run((ROOT / "examples/uc1_left_turn_intersection.json").read_text(), "/tmp/tsf_test_uc1", qc=False)
    return rep._sim, rep._log


@pytest.fixture(scope="module")
def highway():
    from tsf.verify import ladder
    rep = ladder.run((ROOT / "examples/double_cut_in_highway.json").read_text(), "/tmp/tsf_test_hw", qc=False)
    return rep._sim, rep._log


# ------------------------------------------------------------------------------ pure geometry
def test_world_frame_roundtrip():
    wf = rds_hq.WorldFrame(np.array([100.0, -20.0]), math.radians(35))
    assert np.allclose(wf.xy([100.0, -20.0]), 0)
    ahead = np.array([100.0, -20.0]) + 10 * np.array([math.cos(math.radians(35)), math.sin(math.radians(35))])
    assert np.allclose(wf.xy(ahead), [10, 0], atol=1e-9)           # ego heading becomes +x
    left = np.array([100.0, -20.0]) + 5 * np.array([-math.sin(math.radians(35)), math.cos(math.radians(35))])
    assert np.allclose(wf.xy(left), [0, 5], atol=1e-9)             # ego left becomes +y


def test_cuboid_front_normal_is_local_x():
    v = rds_hq._cuboid([10, 5, 3], math.radians(90), (0.2, 0.4, 0.6))
    front = v[[2, 3, 7, 6]].mean(0)                                # face the renderer treats as the front
    assert np.allclose(front - v.mean(0), [0, 0.1, 0], atol=1e-9)  # yaw 90 deg: +x -> +y, half thickness 0.1
    edges = np.stack([v[3] - v[0], v[1] - v[0], v[4] - v[0]])
    assert np.linalg.det(edges) > 0


def test_offset_left_positive():
    pts = np.stack([np.linspace(0, 10, 11), np.zeros(11)], 1)
    assert np.allclose(rds_hq._offset(pts, 1.0)[:, 1], 1.0)


# ------------------------------------------------------------------------------ export against NVIDIA's validator
@needs_nvidia
def test_uc1_export_validates_and_frames(uc1, tmp_path):
    sim, log = uc1
    root = rds_hq.export_clip(sim, log, tmp_path / "rds", "uc1_test", t0=12.0)
    rep = rds_hq.export_report(root, "uc1_test")
    nv = rds_hq.nvidia_io()
    v = nv.validate_clip(root, "uc1_test", "ftheta", require_planar_polygons=True)
    assert v["pose_frames"] == 297 and len(v["cameras"]) == 7
    assert v["tracks"] == 6 and v["map_labels"]["3d_traffic_lights"] == 16          # 4 arms x (main, left) x (near, far)
    v2w = np.load(root / "tsf_export" / "uc1_test.vehicle_to_world.npy")
    assert np.allclose(v2w[0], np.eye(4))                           # rear-axle ground origin, FLU axes at frame 0
    assert rep["ego"]["yaw_change_deg"] == pytest.approx(90.0, abs=1.0)    # the left turn
    # rear axle sits 0.3 * length behind the sim centre: first-frame ego centre is at +0.3L along world x
    e = log.roles.index("ego")
    L = float(log.lwh[e, 0])
    k0 = int(round(12.0 / (log.t[1] - log.t[0])))
    sim_c = log.states[k0, e, :2]
    sim_ra = np.array(rep["ego"]["sim_rear_axle0"])
    assert np.hypot(*(sim_c - sim_ra)) == pytest.approx(0.3 * L, abs=1e-3)


@needs_nvidia
def test_signal_states_follow_log(uc1, tmp_path):
    sim, log = uc1
    root = rds_hq.export_clip(sim, log, tmp_path / "rds", "uc1_sig", t0=12.0)
    import tarfile
    with tarfile.open(root / "3d_traffic_lights_lightnet_status" / "uc1_sig.tar") as tf:
        agg = json.loads(tf.extractfile(tf.getmembers()[0]).read())
    rep = rds_hq.export_report(root, "uc1_sig")
    assert rep["signal_merge_check"]["merge_mismatch_frames"] == 0
    dt = float(log.t[1] - log.t[0])
    for i, h in enumerate(rep["traffic_lights"]):
        st = agg["traffic_light_states"][str(i)]
        assert st["feature_id"] == h["feature_id"] and len(st["state"]) == 297
        if h["kind"] == "left":
            continue
        for k in (0, 100, 296):
            tick = int(round((12.0 + k / 30) / dt))
            assert st["state"][k].split("_")[0] in (log.signals[h["arm"]][tick], "red")


@needs_nvidia
def test_far_side_repeaters_mirror_the_near_heads(uc1, tmp_path):
    sim, log = uc1
    root = rds_hq.export_clip(sim, log, tmp_path / "rds", "uc1_far", t0=12.0)
    import tarfile
    with tarfile.open(root / "3d_traffic_lights_lightnet_status" / "uc1_far.tar") as tf:
        agg = json.loads(tf.extractfile(tf.getmembers()[0]).read())
    states = {v["feature_id"]: v["state"] for v in agg["traffic_light_states"].values()}
    heads = {h["feature_id"]: h for h in rds_hq.export_report(root, "uc1_far")["traffic_lights"]}
    near = [h for h in heads.values() if not h["far_side"]]
    assert near and len(heads) == 2 * len(near)
    for h in near:
        far = heads[h["feature_id"].replace(f"tl_{h['signal_id']}", f"tl_{h['signal_id']}_far", 1)]
        assert far["arm"] == h["arm"] and states[far["feature_id"]] == states[h["feature_id"]]
        assert 20 < math.dist(far["pos_sim"], h["pos_sim"]) < 60                     # across the junction


@needs_nvidia
def test_highway_export_has_no_lights_and_barriers(highway, tmp_path):
    sim, log = highway
    root = rds_hq.export_clip(sim, log, tmp_path / "rds", "hw_test")        # automatic window
    rep = rds_hq.export_report(root, "hw_test")
    assert rep["map_labels"]["3d_traffic_lights"] == 0
    assert rep["map_labels"]["3d_lanelines"] >= 2 and rep["map_labels"]["3d_road_boundaries"] >= 2
    assert rep["t0_auto"] and 0 <= rep["t0_s"] <= 25 - 9.9 + 1e-6
    assert rep["objects"]["tracks"] == 2                                     # the two cutters


def test_choose_t0_covers_the_interaction(highway):
    sim, log = highway
    t0, info = rds_hq.choose_t0(sim, log, 9.9)
    # both scripted cut-ins (12.5 s and 13.5 s, ending at 15.45 / 16.45 s) fall inside the chosen window
    assert t0 <= 12.5 and t0 + 9.9 >= 16.45, (t0, info)

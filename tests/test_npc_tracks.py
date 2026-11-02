import math

import numpy as np
import pytest

from tsf.ingest.npc import NpcConfig, category, ground_plane, lane_index, npc_summary, npc_tracks, stabilize
from tsf.ingest.vipe import ego_track

FPS, T_END, CAM_H = 15.0, 8.0, 1.4
W, H = 424, 240
K = np.array([300.0, 300.0, W / 2, H / 2])
# camera axes (x right, y down, z forward) expressed in the car frame (x forward, y left, z up)
CAM_IN_CAR = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
SKY, BLIP, BUILDING = 9, 7, 8


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


# id -> (phrase, length, width, height, centre x(t), centre y(t), visible span); ego drives 10 m/s along +x
SCENE = {
    1: ("car", 4.5, 1.8, 1.5, lambda t: 27.25 + 8 * t, lambda t: 0 * t, None),                  # leader, 8 m/s
    2: ("car", 4.5, 1.8, 1.5, lambda t: 12.25 + 12 * t, lambda t: 3.5 + 0 * t, None),            # overtakes, left
    3: ("car", 4.5, 1.8, 1.5, lambda t: 42.25 + 0 * t, lambda t: -3.5 + 0 * t, None),            # parked, right
    4: ("truck", 7.0, 2.5, 3.2, lambda t: 35.5 + 9 * t, lambda t: -3.5 + 3.5 * smooth((t - 3) / 3), None),
    BLIP: ("car", 4.5, 1.8, 1.5, lambda t: 30 + 10 * t, lambda t: -7.0 + 0 * t, (2.6, 3.0)),     # 4 frames only
    BUILDING: ("car", 10.0, 10.0, 12.0, lambda t: 60.0 + 0 * t, lambda t: 9.0 + 0 * t, None),   # false positive
}


def pose_at(t, pitch_deg=2.0):
    a = math.radians(pitch_deg)
    tilt = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    T = np.eye(4)
    T[:3, :3] = CAM_IN_CAR @ tilt
    T[:3, 3] = [10.0 * t, 0.0, CAM_H]
    return T


def render(T, t, rng):
    """Depth (H, W) and instance map of the scene seen from camera-to-world pose T at time t."""
    v, u = np.mgrid[0:H, 0:W]
    D = np.stack([(u + 0.5 - K[2]) / K[0], (v + 0.5 - K[3]) / K[1], np.ones((H, W))], -1) @ T[:3, :3].T
    o = T[:3, 3]
    depth = np.zeros((H, W))
    lab = np.zeros((H, W), np.uint8)
    below = D[..., 2] < -1e-6
    depth[below] = -o[2] / D[..., 2][below]
    lab[~below] = SKY
    for k, (_, L, Wd, Hh, fx, fy, span) in SCENE.items():
        if span and not span[0] <= t <= span[1]:
            continue
        c = np.array([fx(t), fy(t), Hh / 2])
        e = np.array([L / 2, Wd / 2, Hh / 2])
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (-e - (o - c)) / D, (e - (o - c)) / D
        tn, tf = np.minimum(t1, t2).max(-1), np.maximum(t1, t2).min(-1)
        hit = (tn <= tf) & (tn > 0) & ((depth == 0) | (tn < depth))
        depth[hit], lab[hit] = tn[hit], k
    return depth, lab


def window(i0, i1, rng, relabel=0, flicker=0.05):
    t = np.arange(i0, i1) / FPS
    depth = np.zeros((len(t), H, W), np.float32)
    inst = np.zeros((len(t), H, W), np.uint8)
    for j, tt in enumerate(t):
        d, lab = render(pose_at(tt), tt, rng)
        depth[j] = d * np.exp(rng.normal(0, 0.03, d.shape)) * (1 + rng.normal(0, flicker))
        inst[j] = np.where(lab > 0, lab + relabel, 0)
    phrases = {k + relabel: v[0] for k, v in SCENE.items()} | {SKY + relabel: "sky"}
    return {"depth": depth, "instance": inst, "K": K, "phrases": phrases,
            "poses": np.stack([pose_at(x) for x in t]), "t": t}


def truth(t):
    """Rear-face centre of the real vehicles (what the bottom band of the mask sees), id -> (N, 2)."""
    return {k: np.stack([fx(t) - L / 2, fy(t)], 1) for k, (_, L, _, _, fx, fy, _) in SCENE.items()
            if k in (1, 2, 3, 4)}


@pytest.fixture(scope="module")
def scene():
    rng = np.random.default_rng(3)
    n = int(T_END * FPS) + 1
    win = window(0, n, rng)
    ego = ego_track(win["poses"], win["t"])
    return win, ego, npc_tracks(win, ego)


def match(tr, gt):
    """Ground-truth id whose rear-face track is nearest to the track's first position."""
    p, i = np.array([tr["x"][0], tr["y"][0]]), round(tr["t"][0] * FPS)
    return min(gt, key=lambda k: np.linalg.norm(gt[k][i] - p))


def test_category_mapping():
    assert category("car") == "car" and category("vehicle") == "car" and category("bus") == "bus"
    assert category("motorcycle") == "motorcycle" and category("person") == "person"
    assert category("sky") is None and category("animal") is None


def test_road_plane_and_scale_stabilisation():
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-6, 6, 4000), np.full(4000, CAM_H), rng.uniform(4, 40, 4000)])
    pts[:, 1] += rng.normal(0, 0.02, 4000)
    pts = np.vstack([pts, np.column_stack([rng.uniform(-6, 6, 800), rng.uniform(-3, 1, 800), rng.uniform(5, 30, 800)])])
    n, h, ratio = ground_plane(pts, np.array([0.0, 1.0, 0.0]), rng)
    assert h == pytest.approx(CAM_H, abs=0.05) and n[1] > 0.99 and ratio > 0.5
    flick = 1 + 0.06 * rng.standard_normal(200)
    s, h_ref = stabilize([(n, CAM_H * f, 0.7) for f in flick])
    assert h_ref == pytest.approx(CAM_H, abs=0.02)
    assert np.std(CAM_H * flick * s) < 0.1 * np.std(CAM_H * flick)


def test_recovers_synthetic_traffic(scene):
    win, _, tracks = scene
    gt = truth(win["t"])
    assert [tr["category"] for tr in sorted(tracks, key=lambda x: x["id"])].count("truck") == 1
    assert len(tracks) == 4, [(tr["category"], round(tr["t0"], 1), round(tr["t1"], 1)) for tr in tracks]
    found = {}
    for tr in tracks:
        k = match(tr, gt)
        found[k] = tr
        i = np.round(tr["t"] * FPS).astype(int)
        err = np.linalg.norm(np.column_stack([tr["x"], tr["y"]]) - gt[k][i], axis=1)
        far = gt[k][i, 0] - 10.0 * tr["t"] > 7.0                      # closer, the car leaves the field of view
        assert np.median(err) < 1.0 and np.percentile(err[far], 90) < 1.5, (k, np.median(err))
        assert tr["confidence"] > 0.3
    assert sorted(found) == [1, 2, 3, 4]
    spd = {1: 8.0, 2: 12.0, 3: 0.0, 4: 9.0}
    for k, tr in found.items():
        mid = (tr["t"] > tr["t"][0] + 0.7) & (tr["t"] < tr["t"][-1] - 0.7)
        assert np.median(tr["speed"][mid]) == pytest.approx(spd[k], abs=1.0), k
    # lanes and relations are relative to the ego path (lane 0 = ego lane, +1 left)
    assert found[1]["lane"][0] == 0 and found[2]["lane"][0] == 1 and found[3]["lane"][0] == -1
    assert (found[1]["relation"], found[2]["relation"], found[3]["relation"]) == (
        "ahead_same_lane", "ahead_left_lane", "parked")
    assert found[1]["gap"][0] == pytest.approx(25.0, abs=1.5) and found[1]["moving"] and not found[3]["moving"]
    # the truck changes from the right lane into the ego lane around t = 4.5 s
    cut = [e for e in found[4]["events"] if e["type"] == "cut_in"]
    assert found[4]["lane"][0] == -1 and found[4]["lane"][-1] == 0
    assert len(cut) == 1 and 3.5 < cut[0]["t"] < 6.0
    summ = npc_summary(tracks)
    assert {s["category"] for s in summ} == {"car", "truck"} and summ[0]["gap0_m"] > 0
    for sm in summ:                      # extrapolated to t = 0: the ego (10 m/s) travels 10 t0, the road user v t0
        assert sm["gap_start_m"] == pytest.approx(sm["gap0_m"] + (10.0 - sm["speed0_kmh"] / 3.6) * sm["t0"], abs=3.0)


def test_blip_building_and_sky_are_dropped(scene):
    _, _, tracks = scene
    assert all(tr["t1"] - tr["t0"] > 1.5 for tr in tracks)
    assert all(tr["size_m"][1] < 4.5 for tr in tracks)                          # no 12 m 'car'
    assert not any(tr["phrase"] == "sky" for tr in tracks)


def test_windows_are_linked_and_ids_do_not_leak():
    rng = np.random.default_rng(5)
    n = int(T_END * FPS) + 1
    w0, w1 = window(0, 70, rng), window(50, n, rng, relabel=20)             # 20 frames overlap, new ids
    ego = ego_track(np.concatenate([w0["poses"], w1["poses"][20:]]), np.arange(n) / FPS)
    tracks = npc_tracks([w0, w1], ego, np.arange(n) / FPS)
    assert len(tracks) == 4
    assert max(tr["t1"] for tr in tracks) > 6.5 and min(tr["t0"] for tr in tracks) < 0.5
    assert all(np.all(np.diff(tr["t"]) > 0) for tr in tracks)


def test_tracks_follow_the_ego_track():
    rng = np.random.default_rng(7)
    n = int(4 * FPS) + 1
    win = window(0, n, rng, flicker=0.0)
    ego = ego_track(win["poses"], win["t"])
    base = npc_tracks(win, ego)
    off = ego.copy()
    off[:, 1] += 2.0                                   # a shifted ego track carries the NPC tracks with it
    moved = npc_tracks(win, off)
    assert moved[0]["y"] - base[0]["y"] == pytest.approx(2.0, abs=1e-6)
    assert moved[0]["lat"] == pytest.approx(base[0]["lat"], abs=0.05)


def test_lane_index_ignores_boundary_noise_and_keeps_real_changes():
    cfg, t = NpcConfig(), np.arange(0, 12, 0.2)
    rng = np.full(len(t), 50.0)
    noisy = 1.6 + 0.5 * np.sin(t * 7.0)                                 # hovers on the 1.75 m lane edge
    assert (lane_index(noisy, rng, t, cfg) == 0).all()
    lat = np.where(t < 6.0, 0.0, 3.5) + 0.3 * np.sin(t * 5.0)           # real lane change at t = 6 s
    lane = lane_index(lat, rng, t, cfg)
    assert lane[0] == 0 and lane[-1] == 1 and (np.diff(lane) != 0).sum() == 1

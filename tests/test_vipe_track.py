import math

import numpy as np
import pytest

from tsf.ingest.vipe import ego_track, metric_scale

V, R_TURN, DT = 10.0, 10.0, 1 / 30
# camera axes (x right, y down, z forward) expressed in the car frame (x forward, y left, z up)
CAM_IN_CAR = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def rot(axis, ang):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def synthetic(pitch_deg=0.0, world=None, scale=1.0):
    """Car: 5 s straight, 90 deg left arc (R=10 m), 5 s straight. Returns camera-to-world poses and times."""
    s_turn = math.pi / 2 * R_TURN
    steps = round((10 + s_turn / V) / DT) + 1
    t = np.arange(steps) * DT
    s = V * t
    s_end = 5 * V + s_turn
    th = np.where(s < 5 * V, 0.0, np.clip((s - 5 * V) / R_TURN, 0, math.pi / 2))
    x = np.where(s < 5 * V, s, 5 * V + R_TURN * np.sin(th) + np.where(s > s_end, s - s_end, 0) * np.cos(th))
    y = np.where(s < 5 * V, 0.0, R_TURN * (1 - np.cos(th)) + np.where(s > s_end, s - s_end, 0) * np.sin(th))
    world = np.eye(3) if world is None else world
    tilt = CAM_IN_CAR @ rot([1, 0, 0], math.radians(pitch_deg))     # pitch down about the camera x axis
    poses = np.tile(np.eye(4), (steps, 1, 1))
    for i in range(steps):
        Rcar = rot([0, 0, 1], th[i])
        poses[i, :3, :3] = world @ Rcar @ tilt
        poses[i, :3, 3] = world @ np.array([x[i], y[i], 0.0]) * scale
    return poses, t, (x[-1], y[-1])


def test_straight_then_left_turn():
    poses, t, (xe, ye) = synthetic()
    tr = ego_track(poses, t)
    assert tr.shape == (len(t), 4)
    assert tr[0] == pytest.approx([0, 0, 0, V], abs=1e-6)
    assert tr[-1, :2] == pytest.approx([xe, ye], abs=1e-6)
    assert (xe, ye) == pytest.approx((5 * V + R_TURN, R_TURN + 5 * V), abs=0.4)  # 50 m, arc, 50 m
    assert tr[-1, 2] == pytest.approx(math.pi / 2, abs=1e-6)          # left turn is positive yaw
    assert tr[:, 2].max() <= math.pi / 2 + 1e-6 and tr[:, 2].min() >= -1e-6
    assert tr[:, 3] == pytest.approx(V, abs=0.15)


def test_gravity_alignment_with_camera_pitch_and_arbitrary_world():
    world = rot([1, 2, 3], 1.1) @ rot([0, 1, 0], 0.4)
    poses, t, _ = synthetic(pitch_deg=6.0, world=world)
    ref = ego_track(synthetic()[0], t)
    tr = ego_track(poses, t)
    assert tr[:, :2] == pytest.approx(ref[:, :2], abs=0.05)
    assert tr[:, 2] == pytest.approx(ref[:, 2], abs=math.radians(1.0))
    assert tr[:, 3] == pytest.approx(V, abs=0.15)


def test_scale_and_frame_time():
    poses, t, (xe, ye) = synthetic(scale=0.5)
    tr = ego_track(poses, t, scale=2.0)
    assert tr[-1, :2] == pytest.approx([xe, ye], abs=1e-6)
    assert tr[:, 3] == pytest.approx(V, abs=0.15)
    per_frame = ego_track(poses)                                      # no timestamps: metres per frame
    assert per_frame[:, 3].mean() == pytest.approx(0.5 * V * DT, abs=0.01)


def test_stationary_start_and_metric_scale():
    poses, t, _ = synthetic()
    poses = np.concatenate([np.repeat(poses[:1], 30, axis=0), poses])
    t = np.arange(len(poses)) * DT
    tr = ego_track(poses, t)
    assert tr[:15, 3] == pytest.approx(0, abs=1e-9)
    assert tr[-1, 2] == pytest.approx(math.pi / 2, abs=1e-6)
    assert metric_scale(0.7) == pytest.approx(2.0) and metric_scale(None) == 1.0
    assert metric_scale(1.4) == pytest.approx(1.0)


def test_despike_removes_glitch_and_keeps_clean_turn():
    from tsf.ingest.vipe import despike
    t = np.arange(0, 20, 1 / 15)
    # slow straight drive (2 m/s) with a SLAM glitch at 8-9 s: +10 m jump forward and a -30 deg heading flip
    # that persists, as on the qdAI dashcam clip
    yaw = np.where(t < 8, 0.0, np.where(t < 9, -math.radians(30) * (t - 8), -math.radians(30)))
    x = 2.0 * t + np.clip(t - 8, 0, 1) * 10.0
    v = np.gradient(x, t)
    tr = despike(np.column_stack([x, np.zeros_like(t), yaw, v]), t)
    assert abs(math.degrees(tr[-1, 2])) < 5
    assert tr[:, 3].max() < 6.0
    assert tr[:, 3][t < 7] == pytest.approx(2.0, abs=1e-6)
    # a real turn (R = 30 m at 8 m/s, 2.1 m/s2 lateral) passes unchanged
    th = 8.0 / 30.0 * t
    clean = np.column_stack([30 * np.sin(th), 30 * (1 - np.cos(th)), th, np.full_like(t, 8.0)])
    out = despike(clean, t)
    assert out[:, 2] == pytest.approx(th, abs=math.radians(0.5))
    assert out[:, :2] == pytest.approx(clean[:, :2], abs=0.3)

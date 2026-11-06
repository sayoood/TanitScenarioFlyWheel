"""Tests for the REFe closed-loop policy wrapper (``tsf.closedloop.refe``): inputs, resampling and the policy contract, with a fake
inference server (no GPU, no network)."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from tsf.closedloop import refe as rf
from tsf.closedloop import rig_adapter as ra

from test_rig_adapter import _syn_rig                     # synthetic Hyperion-like rig


class FakeClient:
    """Stands in for RefeClient: the 'model' drives straight at the ego speed, one proposal."""

    def __init__(self):
        self.calls = []

    def infer(self, img, ego, goal, calib=None, return_all=False):
        self.calls.append((img, np.asarray(ego), np.asarray(goal)))
        v = float(ego[6])
        traj = [[v * 0.2 * (i + 1), 0.0, 0.0] for i in range(20)]
        return {"traj": traj, "k": 3, "agg": 0.9, "latency_ms": {"forward": 1.0, "total": 1.0, "roundtrip": 2.0}}


class StraightPath:
    length = 500.0

    def pose(self, p, d=0.0):
        p = min(max(p, 0.0), self.length)
        return p, 0.0, 0.0


def _ego(**kw):
    d = dict(x=0.0, y=0.0, yaw=0.0, v=10.0, a=0.5, steer=0.05, p=0.0, path=StraightPath())
    d.update(kw)
    return SimpleNamespace(**d)


def _policy(**kw) -> rf.RefePolicy:
    pol = rf.RefePolicy(**kw)
    pol._client = FakeClient()
    return pol


def _navsim_frames(h=512, w=960):
    rng = np.random.default_rng(0)
    return {c: rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for c in ra.NAVSIM_ORDER}


def test_ego_vector_kinematics():
    e = rf.ego_vector(_ego(v=10.0, a=0.5, steer=0.05))
    yaw_rate = 10.0 * math.tan(0.05) / rf.EGO_WHEELBASE
    assert e.shape == (7,) and e.dtype == np.float32
    assert np.allclose(e, [10.0, 0.0, 0.5, 10.0 * yaw_rate, yaw_rate, 0.05, 10.0], atol=1e-5)


def test_route_goal_straight_and_ego_frame():
    # S = max(10, 5) * 12 = 120 m -> goals at 60 m and 120 m along the route
    assert np.allclose(rf.route_goal(_ego()), [60.0, 0.0, 120.0, 0.0])
    # slow ego: the 5 m/s floor applies (S = 60 m)
    assert np.allclose(rf.route_goal(_ego(v=1.0)), [30.0, 0.0, 60.0, 0.0])
    # the ego sits 20 m along, yawed 90 deg left, 1 m off the route: the goals are expressed in ITS frame
    g = rf.route_goal(_ego(x=20.0, y=1.0, yaw=math.pi / 2, p=20.0))
    assert np.allclose(g, [-1.0, -60.0, -1.0, -120.0], atol=1e-9)
    # the route clamps at its end like DriveRL's route_goal_positions
    far = rf.route_goal(_ego(x=450.0, p=450.0))
    assert np.allclose(far, [50.0, 0.0, 50.0, 0.0])
    # no route (T0 smoke tests): straight ahead
    assert np.allclose(rf.route_goal(_ego(path=None)), [60.0, 0.0, 120.0, 0.0])


def test_resample_traj():
    t = np.array([[2.0 * 0.2 * (i + 1), 0.0, 0.0] for i in range(20)])        # 2 m/s straight
    pts = rf.resample_traj(t)
    assert len(pts) == 8 and np.allclose(pts, [[2.0 * 0.5 * (k + 1), 0.0] for k in range(8)])
    # turning rear axle with yaw: the centre-referenced points differ by a (cos yaw - 1) / sin yaw term
    yaw = np.linspace(0.05, 1.0, 20)
    t2 = np.stack([np.linspace(1, 20, 20), np.linspace(0.1, 3, 20), yaw], 1)
    a, b = np.array(rf.resample_traj(t2)), np.array(rf.resample_traj(t2, center_offset_m=1.5))
    assert np.allclose(a[-1] - b[-1], [-1.5 * (math.cos(1.0) - 1.0), -1.5 * math.sin(1.0)], atol=1e-6)
    with pytest.raises(ValueError):
        rf.resample_traj(t, dt=0.5, n=9)


def test_policy_contract_with_navsim_frames():
    pol = _policy()
    out = pol(None, _ego(), _navsim_frames())
    assert pol.name == "refe_sft4" and out["dt"] == 0.5 and len(out["traj"]) == 8
    assert np.allclose(out["traj"], [[10.0 * 0.5 * (k + 1), 0.0] for k in range(8)])
    img, ego7, goal4 = pol._client.calls[0]
    assert img.shape == (4, 512, 960, 3) and img.dtype == np.uint8 and ego7.shape == (7,) and np.allclose(goal4, [60, 0, 120, 0])
    assert pol.last["k"] == 3 and pol.n_calls == 1
    # the closed-loop engine can consume it
    pts = np.asarray(out["traj"])
    assert pts.shape == (8, 2) and np.all(np.diff(pts[:, 0]) > 0)


def test_policy_resizes_navsim_native_frames_and_converts_bgr():
    pol = _policy()
    fr = _navsim_frames(1080, 1920)
    pol(None, _ego(), fr)
    assert pol._client.calls[0][0].shape == (4, 512, 960, 3)
    pol2 = _policy(frames_are_bgr=True)
    fr2 = _navsim_frames()
    pol2(None, _ego(), fr2)
    assert (pol2._client.calls[0][0][0] == fr2["CAM_F0"][..., ::-1]).all()


def test_policy_with_hyperion_frames_goes_through_the_adapter():
    rig = _syn_rig()
    pol = _policy(adapter=ra.RigAdapter(rig))
    frames = {n: np.full((1080, 1920, 3), 90 + i * 10, np.uint8) for i, n in enumerate(rig)}
    out = pol(None, _ego(), frames)
    img = pol._client.calls[0][0]
    assert img.shape == (4, 512, 960, 3) and len(out["traj"]) == 8
    assert img[0].std() < 1.0 and abs(float(img[0].mean()) - 90) < 2          # F0 comes from the front-wide frame (grey 90)


def test_frame_sources_and_missing_frames():
    pol = _policy()
    with pytest.raises(rf.NoFramesError):
        pol(None, _ego())
    pol.set_frames(_navsim_frames())
    assert len(pol(None, _ego())["traj"]) == 8
    pol2 = _policy(frame_source=lambda sim, ego: _navsim_frames())
    assert len(pol2(None, _ego())["traj"]) == 8
    # the T0 smoke-test fallback is explicit, counted, and refuses to run forever
    pol3 = _policy(fallback="const_vel", max_consec_fallback=3)
    out = pol3(None, _ego(v=4.0))
    assert out["traj"][-1] == [16.0, 0.0] and pol3.n_fallback == 1
    pol3(None, _ego())
    with pytest.raises(rf.NoFramesError):
        pol3(None, _ego())


def test_works_as_runner_policy_signature():
    """policy(sim, ego) -> {"traj", "dt"} is all TanitSim._step_external needs."""
    pol = _policy(frame_source=lambda sim, ego: _navsim_frames())
    cmd = pol(object(), _ego())
    assert set(cmd) == {"traj", "dt"} and np.asarray(cmd["traj"]).shape == (8, 2)


def test_replanning_cadence_and_plan_reuse():
    """The model runs at 2 Hz; in between the previous plan is re-expressed in the moved ego frame."""
    pol = _policy()
    fr = _navsim_frames()
    ego = _ego(v=10.0, x=0.0, p=0.0)
    out0 = pol(SimpleNamespace(t=0.0), ego, fr)
    assert len(pol._client.calls) == 1 and not pol.needs_frames(SimpleNamespace(t=0.2)) and pol.needs_frames(SimpleNamespace(t=0.5))
    # 0.2 s later the ego has driven 2 m along the same line: the plan (10 m/s straight) looks the same from the new pose, no frames needed
    ego2 = _ego(v=10.0, x=2.0, p=2.0)
    out1 = pol(SimpleNamespace(t=0.2), ego2)
    assert len(pol._client.calls) == 1
    assert np.allclose(out1["traj"], out0["traj"], atol=1e-6)
    # ... also when the ego has turned: the same world points, new frame
    ego3 = _ego(v=10.0, x=2.0, y=0.0, yaw=0.1, p=2.0)
    p3 = np.asarray(pol(SimpleNamespace(t=0.2), ego3)["traj"])
    w = np.array([[2.0], [0.0]]) + np.array([[math.cos(0.1), -math.sin(0.1)], [math.sin(0.1), math.cos(0.1)]]) @ p3.T
    assert np.allclose(w[0], 10.0 * (0.2 + 0.5 * np.arange(1, 9)), atol=1e-6) and np.allclose(w[1], 0.0, atol=1e-6)
    # at t = 0.5 s the model is queried again
    pol(SimpleNamespace(t=0.5), _ego(x=5.0, p=5.0), fr)
    assert len(pol._client.calls) == 2
    # replan_every_s = 0 -> every call
    pol0 = _policy(replan_every_s=0.0)
    for t in (0.0, 0.05, 0.1):
        pol0(SimpleNamespace(t=t), _ego(), fr)
    assert len(pol0._client.calls) == 3


def test_replan_on_frames_plans_once_per_handed_frame_set():
    """T1 hands frames every 0.533 s: the model runs on each set, with the ego state of the tick they show."""
    pol = _policy(replan_on_frames=True)
    ticks = np.round(np.arange(0.0, 1.6, 0.05), 2)
    for t in ticks:
        if t in (0.0, 0.55, 1.1):
            pol.set_frames(_navsim_frames())
        pol(SimpleNamespace(t=float(t)), _ego(x=10.0 * t, p=10.0 * t))
    assert [c["t"] for c in pol.calls] == [0.0, 0.55, 1.1]

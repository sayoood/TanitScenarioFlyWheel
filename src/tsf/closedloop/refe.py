"""TanitAD REFe SFT-4 (B u400, NAVSIM PDMS 88.25) as the policy under test of the closed loop.

``RefePolicy`` has the closed-loop policy interface of :mod:`tsf.closedloop.runner` (``policy(sim, ego) -> {"traj", "dt"}``,
8 points @ 0.5 s, ego frame x_fwd / y_left) and additionally accepts the camera frames of the current tick:

    policy(sim, ego, frames)                  # frames: {camera name: HxWx3 uint8 RGB}
    policy.frame_source = lambda sim, ego: {...}   # or let the sensor stage (Cosmos T1) hand them over each tick

``frames`` may hold the Hyperion f-theta cameras (``camera_front_wide_120fov`` ... 1920x1080 or 1280x720; they go through
:class:`tsf.closedloop.rig_adapter.RigAdapter` into the 4 NAVSIM pinhole views) or the NAVSIM views themselves
(``CAM_F0 / CAM_L0 / CAM_R0 / CAM_B0``, resized to 960x512). The network runs in a persistent server process
(``deploy/refe/refe_server.py``, torch container on thor3); this module only needs numpy, OpenCV and httpx.

What the model is fed (TanitAD ``refe/planner.py``, the same construction as the NAVSIM evaluation behind the 88.25)
  * 4 images, RGB, CAM_F0 / L0 / R0 / B0, 960x512
  * ego kinematics at the rear axle, body frame: ``vx vy ax ay yaw_rate tire_steering_angle speed``
  * the nav goal: 2 points on the route centre line at arc length S/2 and S ahead of the ego's projection,
    ``S = max(speed, 5 m/s) * 12 s`` (DriveRL ``route_goal_positions``), in the ego frame. Here the route is the closed-loop
    ego path (privileged map access of TanitSim, as the teacher has it); there is NO driving-command input in goal mode.
Output: 64 trajectory proposals x 20 poses @ 5 Hz (x, y, yaw), the proposal with the best predicted aggregate score
(rule ``navsim_v1_lane``, 7-output scorer) is returned, resampled to 8 points @ 0.5 s.
"""
from __future__ import annotations

import io
import math
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from tsf.closedloop import rig_adapter as ra

MODEL_HW = (512, 960)
MODEL_DT = 0.2                      # the model's 20 poses are at 5 Hz, t = 0.2 .. 4.0 s
EGO_WHEELBASE = 2.9                 # TanitSim's kinematic bicycle (engine._step_external)
GOAL_HORIZON_S, GOAL_MIN_SPEED = 12.0, 5.0     # released DriveRL config values used by REFe's planner


class NoFramesError(RuntimeError):
    """The policy was called without camera frames and no fallback is configured."""


# ============================================================================== wire client
class RefeClient:
    """Thin client of ``deploy/refe/refe_server.py`` (npz request, JSON response)."""

    def __init__(self, url: str | None = None, timeout: float = 60.0):
        import httpx
        self.url = (url or os.environ.get("TSF_REFE_URL", "http://127.0.0.1:8765")).rstrip("/")
        self._http = httpx.Client(timeout=timeout)

    def info(self) -> dict:
        return self._http.get(f"{self.url}/info").raise_for_status().json()

    def health(self) -> bool:
        try:
            return bool(self._http.get(f"{self.url}/health", timeout=3.0).json().get("ok"))
        except Exception:
            return False

    def infer(self, img: np.ndarray, ego, goal, calib=None, return_all: bool = False) -> dict:
        buf = io.BytesIO()
        arrs = dict(img=np.ascontiguousarray(img, np.uint8), ego=np.asarray(ego, np.float32), goal=np.asarray(goal, np.float32))
        if calib is not None:
            arrs["calib"] = np.asarray(calib, np.float64)
        if return_all:
            arrs["return_all"] = np.asarray(True)
        np.savez(buf, **arrs)
        t0 = time.perf_counter()
        r = self._http.post(f"{self.url}/infer", content=buf.getvalue(), headers={"content-type": "application/octet-stream"})
        out = r.json()
        if r.status_code != 200:
            raise RuntimeError(f"REFe server refused the request: {out.get('error', r.text)}")
        out["latency_ms"]["roundtrip"] = (time.perf_counter() - t0) * 1e3
        return out


# ============================================================================== inputs from the closed-loop world
def ego_vector(ego, wheelbase: float = EGO_WHEELBASE) -> np.ndarray:
    """``vx vy ax ay yaw_rate tire_steering_angle speed`` (rear axle, body frame) from a TanitSim ``Agent``.

    TanitSim's ego is a kinematic bicycle: no sideslip (vy = 0), yaw rate = v tan(delta) / L, lateral acceleration = v * yaw_rate."""
    v = float(ego.v)
    steer = float(getattr(ego, "steer", 0.0))
    yaw_rate = v * math.tan(steer) / wheelbase
    return np.array([v, 0.0, float(getattr(ego, "a", 0.0)), v * yaw_rate, yaw_rate, steer, v], np.float32)


def to_ego_frame(ego, x: np.ndarray, y: np.ndarray):
    c, s = math.cos(-ego.yaw), math.sin(-ego.yaw)
    dx, dy = np.asarray(x) - ego.x, np.asarray(y) - ego.y
    return c * dx - s * dy, s * dx + c * dy


def route_goal(ego, horizon_s: float = GOAL_HORIZON_S, min_speed: float = GOAL_MIN_SPEED, n: int = 2) -> np.ndarray:
    """``[x1, y1, x2, y2]``: points at arc length S*j/n (j = 1..n) along the ego's route, ego frame (REFe goal input).

    S = max(speed, min_speed) * horizon_s. The route clamps at its end like DriveRL's ``route_goal_positions``;
    without a path (T0 tests) the goals lie straight ahead."""
    S = max(float(ego.v), min_speed) * horizon_s
    path = getattr(ego, "path", None)
    if path is None:
        return np.array([c for j in range(1, n + 1) for c in (S * j / n, 0.0)], np.float32)
    out = []
    for j in range(1, n + 1):
        gx, gy, _ = path.pose(float(ego.p) + S * j / n)
        lx, ly = to_ego_frame(ego, gx, gy)
        out += [float(lx), float(ly)]
    return np.array(out, np.float32)


def resample_traj(traj20: np.ndarray, dt: float = 0.5, n: int = 8, center_offset_m: float = 0.0) -> list:
    """Model output ``[20, 3]`` (x, y, yaw @ 0.2 s steps from t = 0.2, origin = current pose) -> ``n`` (x, y) points @ ``dt``.

    ``center_offset_m`` > 0 re-references the points from the rear axle to the vehicle centre that far ahead of it
    (``center_t = axle_t + R(yaw_t) (a, 0) - (a, 0)``); 0 keeps the rear-axle reference (what TanitSim's controller tracks)."""
    t_src = np.concatenate([[0.0], MODEL_DT * np.arange(1, len(traj20) + 1)])
    p = np.vstack([[0.0, 0.0, 0.0], np.asarray(traj20, float)])
    x, y, yaw = p[:, 0], p[:, 1], np.unwrap(p[:, 2])
    if center_offset_m:
        x, y = x + center_offset_m * (np.cos(yaw) - 1.0), y + center_offset_m * np.sin(yaw)
    t = dt * np.arange(1, n + 1)
    if t[-1] > t_src[-1] + 1e-9:
        raise ValueError(f"requested {t[-1]} s but the model horizon is {t_src[-1]} s")
    return [[float(np.interp(tt, t_src, x)), float(np.interp(tt, t_src, y))] for tt in t]


# ============================================================================== the policy
@dataclass
class RefePolicy:
    """Closed-loop policy wrapper around the REFe server. ``name`` is what ``runner.run`` records."""
    server_url: str | None = None
    adapter: ra.RigAdapter | None = None             # built lazily from the Hyperion rig on the first f-theta frames
    frame_source: object = None                      # optional callable (sim, ego) -> frames
    dt: float = 0.5
    horizon: int = 8
    goal_horizon_s: float = GOAL_HORIZON_S
    goal_min_speed: float = GOAL_MIN_SPEED
    center_offset_m: float = 0.0                     # 0: points are rear-axle referenced (see resample_traj)
    frames_are_bgr: bool = False
    fallback: str = "raise"                          # "raise" | "const_vel" when no frames are available (T0 smoke tests only)
    max_consec_fallback: int = 20
    replan_every_s: float = 0.5                      # TanitSim calls the policy every tick (0.05 s); the model plans at 2 Hz and the
                                                     # plan is re-expressed in the moving ego frame in between (0 = every call)
    name: str = "refe_sft4"
    last: dict = field(default_factory=dict, init=False)
    n_calls: int = field(default=0, init=False)
    n_fallback: int = field(default=0, init=False)
    _consec: int = field(default=0, init=False, repr=False)
    _client: RefeClient | None = field(default=None, init=False, repr=False)
    _latest: dict | None = field(default=None, init=False, repr=False)
    _plan: dict | None = field(default=None, init=False, repr=False)

    # -- frames -------------------------------------------------------------------------------------------------
    @property
    def client(self) -> RefeClient:
        if self._client is None:
            self._client = RefeClient(self.server_url)
        return self._client

    def set_frames(self, frames: dict) -> None:
        """Hand the policy the frames of the current tick (the sensor stage calls this before each ``sim.step``)."""
        self._latest = frames

    def model_images(self, frames: dict) -> np.ndarray:
        """Hyperion f-theta frames or NAVSIM frames -> ``[4, 512, 960, 3]`` uint8 RGB in the REFe camera order."""
        if self.frames_are_bgr:
            frames = {k: np.ascontiguousarray(v[..., ::-1]) for k, v in frames.items()}
        if all(c in frames for c in ra.NAVSIM_ORDER):
            return np.stack([cv2.resize(frames[c], (MODEL_HW[1], MODEL_HW[0])) if frames[c].shape[:2] != MODEL_HW else frames[c]
                             for c in ra.NAVSIM_ORDER], 0)
        if self.adapter is None:
            self.adapter = ra.RigAdapter.from_hyperion()
        return self.adapter.stack(frames)

    def reset(self) -> None:
        """Forget the stored plan (a new scenario; sim time restarts, which ``needs_frames`` also detects)."""
        self._plan = None
        self._consec = 0

    # -- policy interface ------------------------------------------------------------------------------------------
    def needs_frames(self, sim) -> bool:
        """True when the next call will run the model (a sensor stage need only render frames on those ticks)."""
        now = getattr(sim, "t", None)
        return not (self._plan is not None and now is not None and 0.0 <= now - self._plan["t"] < self.replan_every_s - 1e-9)

    def __call__(self, sim, ego, frames: dict | None = None) -> dict:
        if not self.needs_frames(sim):
            return {"traj": self._reuse_plan(sim, ego), "dt": self.dt}
        frames = frames if frames is not None else (self.frame_source(sim, ego) if self.frame_source else self._latest)
        if not frames:
            return self._no_frames(ego)
        self._consec = 0
        t0 = time.perf_counter()
        img = self.model_images(frames)
        t_adapt = (time.perf_counter() - t0) * 1e3
        res = self.client.infer(img, ego_vector(ego), route_goal(ego, self.goal_horizon_s, self.goal_min_speed))
        traj = resample_traj(np.asarray(res["traj"]), self.dt, self.horizon, self.center_offset_m)
        self.n_calls += 1
        res["latency_ms"]["adapter"] = t_adapt
        self.last = {"k": res["k"], "agg": res["agg"], "latency_ms": res["latency_ms"], "traj20": res["traj"]}
        now = getattr(sim, "t", None)
        self._plan = None if now is None else self._store_plan(now, ego, traj)
        return {"traj": traj, "dt": self.dt}

    # -- plan reuse between model calls -----------------------------------------------------------------------------
    def _store_plan(self, now: float, ego, traj: list) -> dict:
        c, s = math.cos(ego.yaw), math.sin(ego.yaw)
        pts = np.vstack([[0.0, 0.0], np.asarray(traj, float)])                       # ego frame at plan time, t = 0, dt, 2dt ...
        wx, wy = ego.x + c * pts[:, 0] - s * pts[:, 1], ego.y + s * pts[:, 0] + c * pts[:, 1]
        return {"t": now, "wx": wx, "wy": wy}

    def _reuse_plan(self, sim, ego) -> list:
        pl = self._plan
        t_rel = (sim.t - pl["t"]) + self.dt * np.arange(1, self.horizon + 1)
        tg = self.dt * np.arange(len(pl["wx"]))
        out = []
        for a in (pl["wx"], pl["wy"]):
            v_end = (a[-1] - a[-2]) / self.dt                                         # beyond the horizon: keep the last velocity
            out.append(np.where(t_rel <= tg[-1], np.interp(t_rel, tg, a), a[-1] + v_end * (t_rel - tg[-1])))
        lx, ly = to_ego_frame(ego, out[0], out[1])
        return [[float(x), float(y)] for x, y in zip(lx, ly)]

    def _no_frames(self, ego) -> dict:
        if self.fallback != "const_vel":
            raise NoFramesError("RefePolicy called without camera frames (pass frames=, set_frames(), or frame_source); "
                                "a camera-only policy must not drive blind. fallback='const_vel' is for T0 smoke tests.")
        self.n_fallback += 1
        self._consec += 1
        if self._consec >= self.max_consec_fallback:
            raise NoFramesError(f"{self._consec} consecutive ticks without frames: the policy never perceived anything")
        v = float(ego.v)
        self.last = {"fallback": True}
        return {"traj": [[v * self.dt * k, 0.0] for k in range(1, self.horizon + 1)], "dt": self.dt}

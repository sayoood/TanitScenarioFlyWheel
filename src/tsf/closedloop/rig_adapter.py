"""Hyperion / PhysicalAI-AV f-theta rig  ->  the 4 NAVSIM pinhole cameras TanitAD REFe expects.

The policy under test (TanitAD REFe SFT-4) consumes ``CAM_F0, CAM_L0, CAM_R0, CAM_B0`` at 960x512. Our sensor stage
(Cosmos, tier T1) and the PhysicalAI-AV recordings deliver the 7 Hyperion f-theta cameras. This module builds, once per
(virtual camera, source camera, frame size), a lookup table (cv2.remap maps + blend weights) that reprojects the f-theta
frames into the NAVSIM cameras, and applies it per frame.

What "the NAVSIM camera" is (taken from TanitAD ``refe/model.py`` ``REFeConfig``, the rig the model was trained under;
``per_sample_calib`` checkpoints are fed exactly these numbers when the rig is the nominal one)
  * intrinsics at the native 1920x1080: fx = fy = 1545, cx = 960, cy = 560 (cy is NOT h/2)
  * the JPEGs are NOT rectified: nuPlan/NAVSIM images carry the Caltech lens distortion
    (k1, k2, p1, p2, k3) = (-0.356123, 0.172545, -0.00213, 0.000464, -0.05231), and the model's 3D position embedding
    UN-distorts every patch ray with a 20 step fixed-point iteration. The virtual camera therefore reproduces the
    distortion, so pixel (u, v) of our output looks along exactly the ray the model assigns to it
    (``distort=False`` gives ideal pinhole images for the ablation ``undistort=False`` only).
  * the model resizes 1920x1080 -> 960x512 (anisotropic, sx = 0.5, sy = 512/1080); the LUT is built directly at the output
    size from the scaled intrinsics, so there is no second interpolation.
  * extrinsics camera -> ego (nuPlan ego = rear axle, x forward, y left, z up): F0 yaw -0.1 deg, L0 +55.2, R0 -55.0, B0 180.

Reprojection (per output pixel)
  1. pixel -> undistorted ray in the virtual camera (same fixed point as the model)
  2. ray -> vehicle frame, intersect with the ground plane z = ``ground_z`` (rays at/above the horizon, or hitting the
     ground beyond ``far_m``, are taken at ``far_m`` = effectively at infinity). This is exact for road surface and moves
     the parallax error of upright objects from ``baseline / depth`` to ``baseline * (1/depth - 1/ground_depth)`` (never worse than pure rotation,
     which is ``ground_plane=False``). The NAVSIM and Hyperion mounts differ by up to ~0.9 m (cross cameras), so this matters.
  3. point -> source camera frame (``camera_to_vehicle`` of the source) -> f-theta projection -> source pixel.
  4. several sources per virtual camera are feathered by their distance to the source image border (``blend_px``);
     sources listed later as a *tier* only fill pixels the earlier tier does not cover.
Pixels no source sees are filled with ``fill`` (ImageNet mean grey, i.e. ~0 after the trunk's normalisation).

Assumptions that are NOT measured (see deploy/refe/README.md): (a) NAVSIM virtual cameras and Hyperion cameras share the
vehicle frame (rear axle, ground-referenced z); (b) the road is planar at ``ground_z`` (TanitSim has no pitch/roll/terrain);
(c) pixel-centre convention differences (<= 0.5 px) are ignored; (d) frames are resizes, not crops, of the calibrated
resolution (1920x1080 PhysicalAI-AV, 1924x1084 hyperion8.json, or the 1280x720 render of the same rig).

Depends on numpy and OpenCV only (no torch, no GPU); pandas only inside :func:`pai_av_rig` (optional helper).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# ============================================================================== NAVSIM virtual cameras
NAVSIM_NATIVE = (1920, 1080)                                   # (w, h) the intrinsics are measured at
NAVSIM_INTRINSICS = dict(fx=1545.0, fy=1545.0, cx=960.0, cy=560.0)
NAVSIM_DISTORTION = (-0.356123, 0.172545, -0.00213, 0.000464, -0.05231)   # k1 k2 p1 p2 k3 (Caltech / OpenCV)
NAVSIM_OUT = (960, 512)                                        # (w, h) REFe input
NAVSIM_ORDER = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")        # REFeConfig.cameras: the model's camera axis
# camera -> ego: translation (m) and quaternion (w, x, y, z), REFeConfig.cam_t / cam_q (nuPlan DB `camera` table)
NAVSIM_EXTRINSICS = {
    "CAM_F0": ((1.671059527953005, -0.015845052723399974, 1.498692086468587),
               (-0.48907127214206575, 0.510302361005295, -0.49953659413553425, 0.5008632370590135)),
    "CAM_L0": ((1.6525236869932511, 0.14301533004884703, 1.5037464761814885),
               (-0.6669006357745871, 0.6825368654798103, -0.20513261807068844, 0.21750305346801996)),
    "CAM_R0": ((1.6252642952324325, -0.16984766018479558, 1.506127136781794),
               (-0.21792717623664432, 0.20689410777832956, -0.67526131123783, 0.6735909259851252)),
    "CAM_B0": ((-0.5014443985418284, 0.01454315307533865, 1.4500732895699808),
               (0.5025840662459249, -0.49753586562491176, -0.5009195148518135, 0.49894584717502716)),
}

# Which f-theta cameras feed which NAVSIM camera (tiers: later tiers only fill what earlier ones do not cover; a tuple is
# a blend group). Derived from the rig geometry (hyperion8.json: yaw / half-FOV): L0 looks 55 deg left with a ~+-35 deg
# (distorted) FOV = azimuth 20..90, wholly inside cross_left_120 (yaw 66, +-61); B0 looks 180 +-35 = 145..215 which only
# the two 70 fov rear cameras (yaw +-150.5, +-36.5) cover together (they overlap by ~14 deg behind the car, blended); the
# rear tele (30 fov) is a fallback for rays at the very edge. The front tele is not used (F0's 63 deg FOV is inside the wide).
DEFAULT_MAPPING = {
    "CAM_F0": ["camera_front_wide_120fov"],
    "CAM_L0": ["camera_cross_left_120fov", "camera_front_wide_120fov"],
    "CAM_R0": ["camera_cross_right_120fov", "camera_front_wide_120fov"],
    "CAM_B0": [("camera_rear_left_70fov", "camera_rear_right_70fov"), "camera_rear_tele_30fov"],
}
IMAGENET_MEAN_RGB = (124, 116, 104)


def quat_to_mat(q) -> np.ndarray:
    """(w, x, y, z) -> 3x3 rotation, identical to ``REFe model._quat_to_mat``."""
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def pose_matrix(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.asarray(t, float)
    return T


# ============================================================================== f-theta source camera
class FThetaCamera:
    """NVIDIA f-theta camera (DriveWorks / Cosmos ``FThetaCamera``), numpy only.

    ``intrinsics`` = ``[cx, cy, width, height, p0..p5, is_bw_poly, c, d, e]`` (14 values, RDS-HQ / hyperion8.json).
    Backward polynomial: ``alpha = p(r)`` with ``r = |A^-1 (pixel - centre)|`` the pixel radius and ``alpha`` the angle
    from the optical axis (rad); ``A = [[c, d], [e, 1]]``. Camera axes are OpenCV (x right, y down, z forward).
    The forward mapping is the numerical inverse of the (monotone) backward polynomial, so no polynomial refit error.
    """

    def __init__(self, intrinsics):
        a = np.asarray(intrinsics, float).reshape(-1)
        if a.size not in (10, 11, 14):
            raise ValueError(f"f-theta intrinsics must have 14 values [cx cy w h p0..p5 is_bw c d e], got {a.size}")
        self.cx, self.cy, self.width, self.height = float(a[0]), float(a[1]), int(round(a[2])), int(round(a[3]))
        self.poly = a[4:10].copy()
        self.is_bw = bool(a[10] > 0) if a.size > 10 else True
        c, d, e = (a[11:14] if a.size >= 14 else (1.0, 0.0, 0.0))
        self.A = np.array([[c, d], [e, 1.0]])
        self.inv_A = np.linalg.inv(self.A)
        if not self.is_bw:
            raise NotImplementedError("forward-polynomial f-theta cameras are not used by the Hyperion / PhysicalAI-AV rigs")
        r_max = math.hypot(max(self.cx, self.width - self.cx), max(self.cy, self.height - self.cy))
        r = np.linspace(0.0, r_max * 1.02, 8192)
        al = self._bw(r)
        if not np.all(np.diff(al) > 0):
            raise ValueError("f-theta backward polynomial is not monotone over the image radius")
        self._r_tab, self._a_tab = r, al
        self.max_angle = float(al[-1])
        self.intrinsics = a

    def _bw(self, r):
        return np.polyval(self.poly[::-1], r)

    @classmethod
    def from_pai_av(cls, width, height, cx, cy, bw_poly):
        """One row of PhysicalAI-AV ``camera_intrinsics`` (bw_poly_0..4; the 6th coefficient is 0)."""
        p = list(bw_poly) + [0.0] * (6 - len(bw_poly))
        return cls([cx, cy, width, height, *p, 1, 1.0, 0.0, 0.0])

    def fov_deg(self) -> float:
        return float(np.degrees(self.max_angle) * 2)

    # -- model --------------------------------------------------------------------------------------------------
    def pixel2ray(self, uv: np.ndarray) -> np.ndarray:
        """[N, 2] pixels (in the calibration resolution) -> [N, 3] unit rays."""
        uv = np.asarray(uv, float).reshape(-1, 2)
        xd = (uv - [self.cx, self.cy]) @ self.inv_A.T
        r = np.hypot(xd[:, 0], xd[:, 1])
        al = self._bw(r)
        s = np.sin(al) / np.maximum(r, 1e-12)
        ray = np.stack([s * xd[:, 0], s * xd[:, 1], np.cos(al)], 1)
        ray[r < 1e-12] = (0.0, 0.0, 1.0)
        return ray

    def ray2pixel(self, rays: np.ndarray):
        """[N, 3] rays (any length) -> ([N, 2] pixels in the calibration resolution, [N] bool angle-in-model-range)."""
        rays = np.asarray(rays, float).reshape(-1, 3)
        nrm = np.linalg.norm(rays, axis=1)
        al = np.arccos(np.clip(rays[:, 2] / np.maximum(nrm, 1e-12), -1.0, 1.0))
        ok = al <= self.max_angle
        r = np.interp(al, self._a_tab, self._r_tab)
        xy = np.hypot(rays[:, 0], rays[:, 1])
        k = np.where(xy > 1e-12, r / np.maximum(xy, 1e-12), 0.0)
        rel = np.stack([k * rays[:, 0], k * rays[:, 1]], 1) @ self.A.T
        return rel + [self.cx, self.cy], ok


# ============================================================================== NAVSIM virtual pinhole camera
@dataclass
class NavsimCamera:
    """Pinhole + Caltech distortion camera at ``out_size`` (intrinsics scaled from ``NAVSIM_NATIVE``)."""
    name: str
    t: tuple
    q: tuple                                   # (w, x, y, z), camera -> ego
    out_size: tuple = NAVSIM_OUT
    fx: float = NAVSIM_INTRINSICS["fx"]
    fy: float = NAVSIM_INTRINSICS["fy"]
    cx: float = NAVSIM_INTRINSICS["cx"]
    cy: float = NAVSIM_INTRINSICS["cy"]
    dist: tuple = NAVSIM_DISTORTION
    native: tuple = NAVSIM_NATIVE
    distort: bool = True
    undistort_iters: int = 20                  # REFeConfig.undistort_iters

    @property
    def R(self) -> np.ndarray:
        return quat_to_mat(self.q)

    @property
    def K(self) -> np.ndarray:
        sx, sy = self.out_size[0] / self.native[0], self.out_size[1] / self.native[1]
        return np.array([[self.fx * sx, 0, self.cx * sx], [0, self.fy * sy, self.cy * sy], [0, 0, 1.0]])

    def dist_coeffs(self) -> np.ndarray:
        return np.array(self.dist if self.distort else (0, 0, 0, 0, 0), float)

    def pixel_rays(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Output pixels -> (xu, yu, 1) in the camera frame; the model's ``_lift_camera`` fixed-point un-distortion."""
        K = self.K
        xd, yd = (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]
        if self.distort:
            k1, k2, p1, p2, k3 = self.dist
            xu, yu = xd.copy(), yd.copy()
            for _ in range(self.undistort_iters):
                r2 = xu * xu + yu * yu
                radial = 1.0 / (1.0 + r2 * (k1 + r2 * (k2 + r2 * k3)))
                dx = 2.0 * p1 * xu * yu + p2 * (r2 + 2.0 * xu * xu)
                dy = p1 * (r2 + 2.0 * yu * yu) + 2.0 * p2 * xu * yu
                xu, yu = (xd - dx) * radial, (yd - dy) * radial
            xd, yd = xu, yu
        return np.stack([xd, yd, np.ones_like(xd)], -1)

    def project(self, pts_cam: np.ndarray) -> np.ndarray:
        """[N, 3] camera-frame points -> [N, 2] output pixels (forward model, via OpenCV ``projectPoints``)."""
        pts = np.asarray(pts_cam, float).reshape(-1, 1, 3)
        uv, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), self.K, self.dist_coeffs())
        return uv.reshape(-1, 2)

    def to_ego(self, pts_cam: np.ndarray) -> np.ndarray:
        return np.asarray(pts_cam, float) @ self.R.T + np.asarray(self.t, float)


def navsim_cameras(out_size=NAVSIM_OUT, distort: bool = True) -> dict[str, NavsimCamera]:
    return {n: NavsimCamera(n, t, q, out_size=tuple(out_size), distort=distort) for n, (t, q) in NAVSIM_EXTRINSICS.items()}


# ============================================================================== rig loading
def load_rig(rig=None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{name: (ftheta_intrinsics[14], camera_to_vehicle[4, 4])}`` from a hyperion8.json-schema file / dict, or (None /
    ``"hyperion"``) via :func:`tsf.cosmos.rds_hq.load_rig` (the framework checkout), the same loader the WSM bridge uses."""
    if rig is None or (isinstance(rig, str) and rig in ("hyperion", "hyperion7", "hyperion8", "hyperion11")):
        from tsf.cosmos.rds_hq import load_rig as _rds_load
        return _rds_load(rig or "hyperion")
    if isinstance(rig, (str, Path)):
        rig = json.loads(Path(rig).read_text())
    if all(isinstance(v, tuple) for v in rig.values()):
        return {k: (np.asarray(v[0], float), np.asarray(v[1], float)) for k, v in rig.items()}
    return {k: (np.asarray(v["ftheta_intrinsics"], float), np.asarray(v["camera_to_vehicle"], float)) for k, v in rig.items()}


def pai_av_rig(clip_id: str, intrinsics_parquet: str, extrinsics_parquet: str, cameras=None) -> dict:
    """Rig of one PhysicalAI-AV clip in the same ``{name: (ftheta14, camera_to_vehicle)}`` form (pandas needed here only).

    Extrinsics rows are ``qx qy qz qw x y z`` (sensor -> vehicle, OpenCV camera axes; verified: front_wide z axis = +x)."""
    import pandas as pd
    from scipy.spatial.transform import Rotation
    ci = pd.read_parquet(intrinsics_parquet).loc[clip_id]
    ex = pd.read_parquet(extrinsics_parquet).loc[clip_id]
    out = {}
    for name in (cameras or [n for n in ci.index if n.startswith("camera_")]):
        r, e = ci.loc[name], ex.loc[name]
        cam = FThetaCamera.from_pai_av(int(r.width), int(r.height), r.cx, r.cy, [r[f"bw_poly_{i}"] for i in range(5)])
        R = Rotation.from_quat([e.qx, e.qy, e.qz, e.qw]).as_matrix()
        out[name] = (cam.intrinsics.copy(), pose_matrix(R, [e.x, e.y, e.z]))
    return out


# ============================================================================== the adapter
@dataclass
class _Lut:
    mapx: np.ndarray                 # float32 [H, W] source x (frame pixels)
    mapy: np.ndarray
    w: np.ndarray                    # float32 [H, W] normalised blend weight of this source (0 where it does not contribute)
    valid_frac: float


@dataclass
class RigAdapter:
    """f-theta frames -> NAVSIM pinhole frames. ``source_rig`` = ``load_rig()`` output.

    ``adapter(frames)`` with ``frames = {source camera name: HxWx3 uint8 RGB}`` returns ``{CAM_F0|L0|R0|B0: 512x960x3 uint8}``
    for every virtual camera whose first-tier source is present; ``stack(frames)`` returns the ``[4, H, W, 3]`` array in the model's
    camera order. Lookup tables are built lazily per (virtual camera, source, frame size) and cached (``cache_dir`` persists them).
    """
    source_rig: dict
    mapping: dict = field(default_factory=lambda: {k: list(v) for k, v in DEFAULT_MAPPING.items()})
    out_size: tuple = NAVSIM_OUT
    distort: bool = True
    ground_plane: bool = True
    ground_z: float = 0.0
    far_m: float = 1.0e4
    blend_px: float = 64.0                    # feather width, in 1920-wide source pixels
    fill: tuple = IMAGENET_MEAN_RGB
    interpolation: int = cv2.INTER_LINEAR
    cache_dir: str | None = None

    def __post_init__(self):
        self.virtual = navsim_cameras(self.out_size, self.distort)
        self.sources = {n: (FThetaCamera(i), np.asarray(T, float)) for n, (i, T) in self.source_rig.items()}
        self._lut: dict = {}
        self._rays: dict = {}
        self._wcache: dict = {}
        for vn, tiers in self.mapping.items():
            for grp in tiers:
                for s in ((grp,) if isinstance(grp, str) else grp):
                    if s not in self.sources:
                        raise KeyError(f"mapping {vn}: source {s!r} not in the rig ({sorted(self.sources)})")

    @classmethod
    def from_hyperion(cls, rig=None, **kw) -> "RigAdapter":
        return cls(load_rig(rig), **kw)

    # -- LUT construction -------------------------------------------------------------------------------------------
    def _vehicle_rays(self, vname: str):
        """Origin O [3] and direction D [H*W, 3] (vehicle frame) of every output pixel; and the ground-plane parameter."""
        if vname not in self._rays:
            cam = self.virtual[vname]
            W, H = self.out_size
            v, u = np.mgrid[0:H, 0:W].astype(np.float64)
            d_cam = cam.pixel_rays(u.reshape(-1), v.reshape(-1))
            D = d_cam @ cam.R.T
            D /= np.linalg.norm(D, axis=1, keepdims=True)                  # unit directions, vehicle frame
            origin = np.asarray(cam.t, float)
            s = np.full(len(D), self.far_m)                                # distance along the ray
            if self.ground_plane:
                down = D[:, 2] < -1e-9
                s_g = np.where(down, (self.ground_z - origin[2]) / np.where(down, D[:, 2], -1.0), np.inf)
                s = np.minimum(np.maximum(s_g, 0.0), self.far_m)           # horizon / far rays end at far_m (~ infinity)
            self._rays[vname] = origin + D * s[:, None]
        return self._rays[vname]

    def _build(self, vname: str, sname: str, frame_wh: tuple) -> _Lut:
        key = (vname, sname, tuple(frame_wh))
        if key in self._lut:
            return self._lut[key]
        path = None
        if self.cache_dir:
            tag = f"{vname}__{sname}__{frame_wh[0]}x{frame_wh[1]}__{self.out_size[0]}x{self.out_size[1]}__{int(self.distort)}{int(self.ground_plane)}.npz"
            path = Path(self.cache_dir) / tag
            if path.exists():
                z = np.load(path)
                self._lut[key] = _Lut(z["mapx"], z["mapy"], z["w"], float(z["valid_frac"]))
                return self._lut[key]
        cam, T = self.sources[sname]
        P = self._vehicle_rays(vname)
        p_src = (P - T[:3, 3]) @ T[:3, :3]                 # R^T (P - t)
        uv, ok = cam.ray2pixel(p_src)
        Wf, Hf = frame_wh
        uvf = uv * [Wf / cam.width, Hf / cam.height]       # calibration pixels -> frame pixels
        inside = ok & (uvf[:, 0] >= 0) & (uvf[:, 0] <= Wf - 1) & (uvf[:, 1] >= 0) & (uvf[:, 1] <= Hf - 1)
        border = np.minimum.reduce([uvf[:, 0], Wf - 1 - uvf[:, 0], uvf[:, 1], Hf - 1 - uvf[:, 1]]) * (1920.0 / Wf)
        H, W = self.out_size[1], self.out_size[0]
        lut = _Lut(uvf[:, 0].reshape(H, W).astype(np.float32), uvf[:, 1].reshape(H, W).astype(np.float32),
                   np.where(inside, np.maximum(border, 1e-3), 0.0).reshape(H, W).astype(np.float32), float(inside.mean()))
        self._lut[key] = lut                               # `w` holds the raw border distance until `_weights` normalises it
        if path is not None:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            np.savez(path, mapx=lut.mapx, mapy=lut.mapy, w=lut.w, valid_frac=lut.valid_frac)
        return lut

    def _weights(self, vname: str, frame_wh: dict) -> tuple:
        """([(source, lut, normalised weight [H, W])], covered [H, W] bool) over all tiers (an earlier tier wins where it covers)."""
        key = (vname, tuple((n, tuple(frame_wh[n])) for grp in self.mapping[vname]
                            for n in ((grp,) if isinstance(grp, str) else grp) if n in frame_wh))
        if key in self._wcache:
            return self._wcache[key]
        out, covered = [], None
        for grp in self.mapping[vname]:
            names = (grp,) if isinstance(grp, str) else tuple(grp)
            names = [n for n in names if n in frame_wh]
            if not names:
                continue
            luts = [self._build(vname, n, frame_wh[n]) for n in names]
            d = np.stack([lt.w for lt in luts])                                  # border distance, 0 = invalid
            dmax = d.max(0)
            w = np.where(d > 0, np.clip(1.0 - (dmax - d) / max(self.blend_px, 1e-6), 0.0, 1.0), 0.0)
            tot = w.sum(0)
            wn = np.where(tot > 0, w / np.maximum(tot, 1e-9), 0.0)
            if covered is not None:
                wn = wn * (~covered)[None]
            for n, lt, wi in zip(names, luts, wn):
                out.append((n, lt, wi.astype(np.float32)))
            has = wn.sum(0) > 0
            covered = has if covered is None else (covered | has)
        res = (out, (covered if covered is not None else np.zeros(self.out_size[::-1], bool)))
        self._wcache[key] = res
        return res

    # -- application -----------------------------------------------------------------------------------------------
    def virtual_frame(self, vname: str, frames: dict) -> np.ndarray:
        """One NAVSIM camera image (uint8 RGB, ``out_size``) from the available source frames."""
        fwh = {n: (f.shape[1], f.shape[0]) for n, f in frames.items() if n in self.sources}
        parts, covered = self._weights(vname, fwh)
        if not parts:
            raise KeyError(f"{vname}: none of its sources {self.mapping[vname]} is in the frames {sorted(frames)}")
        acc = np.zeros((self.out_size[1], self.out_size[0], 3), np.float32)
        for n, lt, w in parts:
            if not w.any():
                continue
            img = cv2.remap(np.ascontiguousarray(frames[n]), lt.mapx, lt.mapy, self.interpolation,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(int(c) for c in self.fill))
            acc += img.astype(np.float32) * w[..., None]
        fill = np.asarray(self.fill, np.float32)
        acc += (~covered)[..., None] * fill
        return np.clip(acc + 0.5, 0, 255).astype(np.uint8)

    def __call__(self, frames: dict) -> dict:
        return {v: self.virtual_frame(v, frames) for v in NAVSIM_ORDER}

    def stack(self, frames: dict) -> np.ndarray:
        """``[4, H, W, 3]`` uint8 in the REFe camera order (F0, L0, R0, B0)."""
        out = self(frames)
        return np.stack([out[v] for v in NAVSIM_ORDER], 0)

    def coverage(self, frame_wh: dict) -> dict:
        """Fraction of output pixels each virtual camera gets from at least one source, for the given frame sizes."""
        res = {}
        for v in NAVSIM_ORDER:
            _, cov = self._weights(v, frame_wh)
            res[v] = float(cov.mean())
        return res


def sources_needed(mapping: dict | None = None) -> list[str]:
    m = mapping or DEFAULT_MAPPING
    names = []
    for tiers in m.values():
        for grp in tiers:
            for s in ((grp,) if isinstance(grp, str) else grp):
                if s not in names:
                    names.append(s)
    return names

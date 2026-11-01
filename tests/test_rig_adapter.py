"""Tests for the f-theta -> NAVSIM pinhole rig adapter (``tsf.closedloop.rig_adapter``). numpy + OpenCV only, no GPU.

The rig used here is a SYNTHETIC Hyperion-like rig (mount positions / yaws and f-theta polynomials in the range of
NVIDIA's hyperion8.json and the PhysicalAI-AV calibration); the real ``hyperion8.json`` is only used when the Cosmos checkout is present.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from tsf.closedloop import rig_adapter as ra

# name: (yaw deg, pitch deg, (x, y, z), poly p1..p4)  -- values of the order of hyperion8.json / PhysicalAI-AV
_SYN = {
    "camera_front_wide_120fov": (-0.5, -0.3, (1.75, 0.0, 1.45), (1.0821e-3, 6.4e-9, 1.8e-11, 1.4e-15)),
    "camera_cross_left_120fov": (66.0, -0.2, (2.51, 0.96, 0.90), (1.0821e-3, -3.6e-10, 3.07e-11, -6.26e-15)),
    "camera_cross_right_120fov": (-66.7, 1.3, (2.52, -0.94, 0.90), (1.0814e-3, -2.9e-11, 2.98e-11, -5.17e-15)),
    "camera_rear_left_70fov": (150.9, -2.9, (1.92, 1.05, 0.90), (6.42e-4, 4.24e-9, -1.5e-11, 1.226e-14)),
    "camera_rear_right_70fov": (-150.2, -0.7, (1.92, -1.03, 0.92), (6.43e-4, 3.48e-9, -1.44e-11, 1.233e-14)),
    "camera_rear_tele_30fov": (-180.0, 2.0, (0.48, 0.32, 1.36), (2.71e-4, 1.17e-10, -1.39e-12, 1.0e-15)),
}


def _cam_to_vehicle(yaw_deg: float, pitch_deg: float, t) -> np.ndarray:
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    z = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
    x = np.cross(z, [0, 0, 1.0])                  # right = forward x up  (OpenCV: x right, y down, z forward)
    x /= np.linalg.norm(x)
    yv = np.cross(z, x)
    return ra.pose_matrix(np.stack([x, yv, z], 1), t)


def _syn_rig(size=(1920, 1080)) -> dict:
    rig = {}
    for n, (yaw, pitch, t, poly) in _SYN.items():
        w, h = size
        intr = [w / 2 - 3.0, h / 2 + 5.0, w, h, 0.0, *poly, 0.0, 1, 1.0, 0.0, 0.0]
        rig[n] = (np.array(intr, float), _cam_to_vehicle(yaw, pitch, t))
    return rig


@pytest.fixture(scope="module")
def rig():
    return _syn_rig()


# ------------------------------------------------------------------------------------------------ camera models
def test_ftheta_roundtrip(rig):
    for n, (intr, _) in rig.items():
        cam = ra.FThetaCamera(intr)
        rng = np.random.default_rng(1)
        uv = rng.uniform([40, 40], [cam.width - 40, cam.height - 40], (500, 2))
        ray = cam.pixel2ray(uv)
        assert np.allclose(np.linalg.norm(ray, axis=1), 1.0, atol=1e-9)
        back, ok = cam.ray2pixel(ray * 7.3)             # length must not matter
        assert ok.all() and np.abs(back - uv).max() < 5e-3, n
    # the optical axis maps to the principal point and the FOV is the advertised one (120 / 70 / 30 deg class)
    fw, rl, rt = (ra.FThetaCamera(rig[n][0]) for n in ("camera_front_wide_120fov", "camera_rear_left_70fov", "camera_rear_tele_30fov"))
    assert np.allclose(fw.ray2pixel(np.array([[0, 0, 1.0]]))[0][0], [fw.cx, fw.cy])
    assert 2 * 55 < fw.fov_deg() < 2 * 75 and rl.fov_deg() < fw.fov_deg() and rt.fov_deg() < rl.fov_deg()


def test_navsim_model_matches_opencv():
    """The fixed-point un-distortion of the REFe model == OpenCV's undistortPoints, and projectPoints inverts it."""
    cam = ra.navsim_cameras()["CAM_F0"]
    v, u = np.mgrid[0:512:37, 0:960:41].astype(np.float64)
    ours = cam.pixel_rays(u.ravel(), v.ravel())
    pts = np.stack([u.ravel(), v.ravel()], 1).reshape(-1, 1, 2)
    ref = cv2.undistortPoints(pts, cam.K, cam.dist_coeffs(), None, None, None,
                              (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-14)).reshape(-1, 2)
    assert np.abs(ours[:, :2] - ref).max() < 2e-5
    assert np.abs(cam.project(ours) - pts.reshape(-1, 2)).max() < 5e-3
    # the intrinsics are the REFe ones scaled 1920x1080 -> 960x512 (cy = 560 is NOT h/2)
    assert np.allclose(np.diag(cam.K)[:2], [772.5, 1545 * 512 / 1080]) and np.isclose(cam.K[1, 2], 560 * 512 / 1080)


def test_virtual_rig_geometry():
    cams = ra.navsim_cameras()
    yaw = {n: math.degrees(math.atan2(c.R[1, 2], c.R[0, 2])) for n, c in cams.items()}
    assert abs(yaw["CAM_F0"]) < 1 and 54 < yaw["CAM_L0"] < 56 and -56 < yaw["CAM_R0"] < -54 and abs(abs(yaw["CAM_B0"]) - 180) < 1
    for c in cams.values():
        assert np.allclose(c.R @ c.R.T, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(c.R), 1.0)


# ------------------------------------------------------------------------------------------------ LUT <-> point agreement
def _bilinear(m: np.ndarray, uv: np.ndarray) -> np.ndarray:
    x0, y0 = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
    fx, fy = uv[:, 0] - x0, uv[:, 1] - y0
    return (m[y0, x0] * (1 - fx) * (1 - fy) + m[y0, x0 + 1] * fx * (1 - fy)
            + m[y0 + 1, x0] * (1 - fx) * fy + m[y0 + 1, x0 + 1] * fx * fy)


def _lut_vs_point(ad: ra.RigAdapter, vname: str, sname: str, P: np.ndarray):
    """pixel of world point P in the virtual cam, LUT value there, and the true f-theta pixel of P in the source."""
    vc = ad.virtual[vname]
    p_cam = (P - np.asarray(vc.t)) @ vc.R                        # R^T (P - t)
    uv_v = vc.project(p_cam)
    lut = ad._build(vname, sname, (1920, 1080))
    got = np.stack([_bilinear(lut.mapx, uv_v), _bilinear(lut.mapy, uv_v)], 1)
    cam, T = ad.sources[sname]
    true, ok = cam.ray2pixel((P - T[:3, 3]) @ T[:3, :3])
    assert ok.all()
    return uv_v, got, true


@pytest.mark.parametrize("vname,sname,pts", [
    ("CAM_F0", "camera_front_wide_120fov", [(12, 1.0), (25, -3.0), (45, 6.0), (8, -2.5), (70, 0.5)]),
    ("CAM_L0", "camera_cross_left_120fov", [(14, 9.0), (22, 16.0), (30, 30.0), (9, 8.0)]),
    ("CAM_R0", "camera_cross_right_120fov", [(14, -9.0), (22, -16.0), (30, -30.0), (9, -8.0)]),
    ("CAM_B0", "camera_rear_left_70fov", [(-12, 2.0), (-25, 6.0), (-40, 9.0)]),
    ("CAM_B0", "camera_rear_right_70fov", [(-12, -2.0), (-25, -6.0), (-40, -9.0)]),
])
def test_ground_points_agree_between_models(rig, vname, sname, pts):
    """A road point (z = 0) must land on the same source pixel the LUT sends its virtual pixel to (ground-plane mode)."""
    ad = ra.RigAdapter(rig)
    P = np.array([[x, y, 0.0] for x, y in pts])
    uv_v, got, true = _lut_vs_point(ad, vname, sname, P)
    in_img = (uv_v[:, 0] > 2) & (uv_v[:, 0] < 957) & (uv_v[:, 1] > 2) & (uv_v[:, 1] < 509)
    assert in_img.any(), "test points are not visible in the virtual camera"
    err = np.abs(got - true)[in_img]
    assert err.max() < 0.15, (vname, sname, err)        # px at 1920x1080 source resolution


def test_far_points_agree_without_ground_plane(rig):
    """At infinity the two cameras' offsets vanish: rotation-only reprojection (ground_plane=False) is exact."""
    ad = ra.RigAdapter(rig, ground_plane=False)
    dirs = np.array([[1, 0.05, 0.02], [1, 0.3, 0.1], [1, -0.2, 0.15], [1, 0.1, -0.05]], float)
    P = dirs / np.linalg.norm(dirs, axis=1, keepdims=True) * 9.0e3
    uv_v, got, true = _lut_vs_point(ad, "CAM_F0", "camera_front_wide_120fov", P)
    assert np.abs(got - true).max() < 0.3


def test_ground_plane_never_worse_than_rotation_only(rig):
    """An upright object (point 1.2 m above the road, 10 m ahead): the ground model reduces the parallax error."""
    P = np.array([[10.0, 1.5, 1.2], [18.0, -2.0, 1.0], [10.0, 6.0, 1.5]])
    err = {}
    for gp in (True, False):
        ad = ra.RigAdapter(rig, ground_plane=gp)
        _, got, true = _lut_vs_point(ad, "CAM_F0", "camera_front_wide_120fov", P)
        err[gp] = np.linalg.norm(got - true, axis=1)
    assert (err[True] <= err[False] + 1e-6).all() and err[True].sum() < err[False].sum()   # ties at the horizon ray


# ------------------------------------------------------------------------------------------------ whole-image test
def _scene_colour(origin: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Analytic scene: smooth textured ground plane z = 0 that melts into the sky colour at the horizon, plus a sky gradient.
    D unit directions, [N, 3] -> [N] grey (0..255). Continuous everywhere, so a sub-pixel misplacement costs a few grey levels."""
    dz = D[:, 2]
    down = dz < -1e-6
    s = np.where(down, -origin[2] / np.where(down, dz, -1.0), 1e9)
    x, y = origin[0] + D[:, 0] * np.minimum(s, 1e5), origin[1] + D[:, 1] * np.minimum(s, 1e5)
    sky = 0.75 + 0.2 * np.clip(dz, 0, 1) + 0.1 * np.sin(3 * np.arctan2(D[:, 1], D[:, 0]))
    sky_h = 0.75 + 0.1 * np.sin(3 * np.arctan2(y, x))                               # the sky at the horizon, as seen from the ground point
    fade = np.exp(-np.hypot(x, y) / 60.0)               # a property of the world point, not of the viewing camera
    tex = 0.30 * np.sin(2 * np.pi * x / 14.0) * np.sin(2 * np.pi * y / 11.0) + 0.1 * np.sin(2 * np.pi * x / 40.0) - 0.30
    ground = sky_h + tex * fade
    return np.clip(np.where(down, ground, sky), 0, 1) * 255.0


def _render_source(cam: ra.FThetaCamera, T: np.ndarray) -> np.ndarray:
    v, u = np.mgrid[0:cam.height, 0:cam.width].astype(np.float64)
    ray = cam.pixel2ray(np.stack([u.ravel(), v.ravel()], 1))
    g = _scene_colour(T[:3, 3], ray @ T[:3, :3].T).reshape(cam.height, cam.width)
    return np.repeat(g[..., None], 3, 2).astype(np.uint8)


def _render_virtual(vc: ra.NavsimCamera) -> np.ndarray:
    W, H = vc.out_size
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    d = vc.pixel_rays(u.ravel(), v.ravel()) @ vc.R.T
    g = _scene_colour(np.asarray(vc.t), d / np.linalg.norm(d, axis=1, keepdims=True)).reshape(H, W)
    return np.repeat(g[..., None], 3, 2).astype(np.uint8)


@pytest.fixture(scope="module")
def synth(rig):
    frames = {n: _render_source(ra.FThetaCamera(i), T) for n, (i, T) in rig.items()}
    ad = ra.RigAdapter(rig)
    return ad, frames, ad(frames)


@pytest.mark.parametrize("vname", ra.NAVSIM_ORDER)
def test_adapter_image_matches_direct_render(synth, vname):
    """The remapped image equals what the NAVSIM camera would have seen of the same (planar-world) scene."""
    ad, frames, out = synth
    ref = _render_virtual(ad.virtual[vname]).astype(np.float32)
    got = out[vname].astype(np.float32)
    assert got.shape == (512, 960, 3) and out[vname].dtype == np.uint8
    covered = np.zeros(got.shape[:2], bool)
    _, covered = ad._weights(vname, {n: (f.shape[1], f.shape[0]) for n, f in frames.items()})
    # B0 loses its top rows (elevation above ~18 deg: the 70 fov rear cameras do not see it) -> grey fill, not compared
    assert covered.mean() > (0.90 if vname == "CAM_B0" else 0.97), f"{vname}: only {covered.mean():.3f} seen by a source"
    d = np.abs(got - ref).mean(-1)[covered]
    assert d.mean() < 1.5 and np.percentile(d, 99) < 8.0, (vname, d.mean(), np.percentile(d, 99))


def test_stack_order_and_uncovered_fill(synth, rig):
    ad, frames, out = synth
    st = ad.stack(frames)
    assert st.shape == (4, 512, 960, 3) and st.dtype == np.uint8
    assert all((st[i] == out[v]).all() for i, v in enumerate(ra.NAVSIM_ORDER))
    # remove the only sources of CAM_B0's first tier -> the rear tele fills only its own 30 deg, the rest is the fill colour
    only_tele = {"camera_rear_tele_30fov": frames["camera_rear_tele_30fov"]}
    b0 = ad.virtual_frame("CAM_B0", only_tele)
    assert (b0 == np.array(ra.IMAGENET_MEAN_RGB, np.uint8)).all(-1).mean() > 0.4
    with pytest.raises(KeyError):
        ad.virtual_frame("CAM_B0", {"camera_front_wide_120fov": frames["camera_front_wide_120fov"]})


def test_lut_cached_and_resolution_independent(rig):
    ad = ra.RigAdapter(rig)
    f1080 = {n: _render_source(ra.FThetaCamera(i), T) for n, (i, T) in rig.items() if n == "camera_front_wide_120fov"}
    ad.virtual_frame("CAM_F0", f1080)
    n_luts = len(ad._lut)
    ad.virtual_frame("CAM_F0", f1080)
    assert len(ad._lut) == n_luts                                  # no rebuild
    # the same scene rendered at 1280x720 (a resize of the calibrated frame) gives the same virtual image
    f720 = {k: cv2.resize(v, (1280, 720), interpolation=cv2.INTER_AREA) for k, v in f1080.items()}
    a, b = ad.virtual_frame("CAM_F0", f1080).astype(float), ad.virtual_frame("CAM_F0", f720).astype(float)
    assert np.abs(a - b).mean() < 2.5 and len(ad._lut) == n_luts + 1


def test_mapping_validation(rig):
    with pytest.raises(KeyError):
        ra.RigAdapter({k: v for k, v in rig.items() if k != "camera_rear_tele_30fov"})
    assert set(ra.sources_needed()) == set(_SYN)


def test_real_hyperion_rig_if_available():
    try:
        rig = ra.load_rig("hyperion")
    except Exception:
        pytest.skip("Cosmos WSM checkout (hyperion8.json) not available")
    ad = ra.RigAdapter(rig)
    cov = ad.coverage({n: (1920, 1080) for n in ra.sources_needed()})
    assert all(c > 0.97 for k, c in cov.items() if k != "CAM_B0") and cov["CAM_B0"] > 0.90, cov

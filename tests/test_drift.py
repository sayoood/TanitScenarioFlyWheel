import cv2
import numpy as np

from tsf.cosmos.drift import DriftGuard, saturation, sharpness


def _scene(seed: int = 0) -> np.ndarray:
    """A textured road-like frame: noise texture, lane stripes and coloured blocks (480x832 BGR)."""
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(40, 200, (480, 832, 3), dtype=np.uint8), (0, 0), 1.0)
    for x in range(60, 832, 120):
        cv2.line(img, (x, 479), (416, 240), (230, 230, 230), 3)
    cv2.rectangle(img, (500, 260), (620, 340), (40, 40, 200), -1)
    cv2.rectangle(img, (150, 280), (260, 360), (200, 120, 30), -1)
    return img


def _drift(img: np.ndarray, sigma: float, sat: float) -> np.ndarray:
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (0, 0), sigma), cv2.COLOR_BGR2HSV).astype(np.float64)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def test_guard_pulls_colour_and_sharpness_back_to_the_reference(tmp_path):
    ref, bad = _scene(), _drift(_scene(), sigma=1.6, sat=1.5)
    cv2.imwrite(str(tmp_path / "ref.png"), ref)
    cv2.imwrite(str(tmp_path / "bad.png"), bad)
    g = DriftGuard()
    g.reference({"front": tmp_path / "ref.png"})
    out, st = g({"front": tmp_path / "bad.png"}, tmp_path / "cond")
    fixed = cv2.imread(str(out["front"]))
    assert sharpness(fixed) > 1.5 * sharpness(bad) and st["front"]["amount"] > 0
    assert abs(saturation(fixed) - saturation(ref)) < 0.5 * abs(saturation(bad) - saturation(ref))


def test_guard_reanchors_a_washed_out_chain_but_not_too_often(tmp_path):
    cv2.imwrite(str(tmp_path / "ref.png"), _scene())
    cv2.imwrite(str(tmp_path / "mush.png"), _drift(_scene(), sigma=9.0, sat=1.0))
    g = DriftGuard(min_gap=2)
    g.reference({"front": tmp_path / "ref.png"})
    calls = [g({"front": tmp_path / "mush.png"}, tmp_path / "c")[0] for _ in range(4)]
    assert [c is None for c in calls] == [False, True, False, True]          # every min_gap-th call at most
    never = DriftGuard(reanchor_below=0.0)
    never.reference({"front": tmp_path / "ref.png"})
    assert all(never({"front": tmp_path / "mush.png"}, tmp_path / "c")[0] is not None for _ in range(4))

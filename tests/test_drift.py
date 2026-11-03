import cv2
import numpy as np

from tsf.cosmos.drift import DriftGuard, box_cover, detail, detail_tiles, saturation, sharpness, veil, veil_tiles


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
    never = DriftGuard(reanchor_below=0.0, veil_below=0.0)
    never.reference({"front": tmp_path / "ref.png"})
    assert all(never({"front": tmp_path / "mush.png"}, tmp_path / "c")[0] is not None for _ in range(4))


def _veiled(img: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    """A brown smear over the lower centre and the road band, as the 720p chain grew it (sky and sides stay sharp)."""
    out = img.copy()
    roi = out[150:400, 120:760]
    out[150:400, 120:760] = cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, (70, 105, 140)), alpha, 0)
    return out


def test_veil_triggers_a_reanchor_that_whole_frame_sharpness_misses(tmp_path):
    ref, smear = _scene(), _veiled(_scene())
    assert sharpness(smear) > 0.3 * sharpness(ref)                          # the sharpness rule alone would keep it
    assert veil(smear, veil_tiles(ref)) < 0.5 and veil(_drift(ref, 0.6, 1.0), veil_tiles(ref)) > 0.75
    cv2.imwrite(str(tmp_path / "ref.png"), ref)
    cv2.imwrite(str(tmp_path / "smear.png"), smear)
    g = DriftGuard(min_gap=1)
    g.reference({"front": tmp_path / "ref.png"})
    out, st = g({"front": tmp_path / "smear.png"}, tmp_path / "c")
    assert out is None and st["reanchor"] == ["front"] and st["front"]["veil"] < 0.5
    off = DriftGuard(min_gap=1, veil_below=0.0)
    off.reference({"front": tmp_path / "ref.png"})
    assert off({"front": tmp_path / "smear.png"}, tmp_path / "c")[0] is not None


def test_a_near_vehicle_is_no_veil_once_its_box_is_masked(tmp_path):
    """A truck side filling the left half of the band (flat, little local contrast) read as a veil and re-anchored the
    chain; with the sim's box for it the guard compares only the uncovered tiles."""
    ref, near = _scene(), _scene()
    cv2.rectangle(near, (0, 120), (430, 420), (170, 175, 180), -1)
    full = veil(near, veil_tiles(ref))
    cover = box_cover([[0, 120, 430, 420]], (832, 480))
    assert full < 0.75 and veil(near, veil_tiles(ref), cover < 0.25) > 0.9
    assert veil(near, veil_tiles(ref), np.zeros((4, 8), bool)) == 1.0       # nothing left to judge
    cv2.imwrite(str(tmp_path / "ref.png"), ref)
    cv2.imwrite(str(tmp_path / "near.png"), near)
    g = DriftGuard(min_gap=1)
    g.reference({"front": tmp_path / "ref.png"})
    boxes = {"front": {"size": [832, 480], "boxes": [[0, 120, 430, 420]]}}
    out, st = g({"front": tmp_path / "near.png"}, tmp_path / "c", boxes=boxes)
    assert out is not None and st["front"]["veil_tiles"] == 16
    assert g({"front": tmp_path / "near.png"}, tmp_path / "c")[0] is None    # without the box: the old false re-anchor


def test_veil_reanchors_only_for_the_cameras_it_is_trusted_on(tmp_path):
    cv2.imwrite(str(tmp_path / "ref.png"), _scene())
    cv2.imwrite(str(tmp_path / "smear.png"), _veiled(_scene()))
    g = DriftGuard(min_gap=1, veil_cams=("front",))
    g.reference({"cross": tmp_path / "ref.png", "front": tmp_path / "ref.png"})
    out, st = g({"cross": tmp_path / "smear.png", "front": tmp_path / "ref.png"}, tmp_path / "c")
    assert out is not None and st["cross"]["veil"] < 0.5                    # reported, not acted on
    assert g({"cross": tmp_path / "ref.png", "front": tmp_path / "smear.png"}, tmp_path / "c")[0] is None


def _shaded(img: np.ndarray) -> np.ndarray:
    """The road turned into smooth shading, as the protected-left chain did it: as much contrast per tile, no texture."""
    out = img.copy()
    y, x = np.mgrid[150:400, 0:832]
    out[150:400] = (120 + 60 * np.sin(2 * np.pi * (x + y) / 120)).astype(np.uint8)[..., None]
    return out


def test_detail_catches_a_road_turned_to_shading_that_the_veil_misses(tmp_path):
    ref, shaded = _scene(), _shaded(_scene())
    assert veil(shaded, veil_tiles(ref)) > 0.75 and detail(shaded, detail_tiles(ref)) < 0.3
    assert detail(_drift(ref, 0.6, 1.0), detail_tiles(ref)) > 0.65                 # a slightly soft frame passes
    cv2.imwrite(str(tmp_path / "ref.png"), ref)
    cv2.imwrite(str(tmp_path / "shaded.png"), shaded)
    g = DriftGuard(min_gap=1, detail_below=0.65)
    g.reference({"front": tmp_path / "ref.png"})
    out, st = g({"front": tmp_path / "shaded.png"}, tmp_path / "c")
    assert out is None and st["reanchor"] == ["front"] and st["front"]["detail"] < 0.3
    off = DriftGuard(min_gap=1)
    off.reference({"front": tmp_path / "ref.png"})
    assert off({"front": tmp_path / "shaded.png"}, tmp_path / "c")[0] is not None


def test_after_a_reanchor_the_structure_is_judged_against_the_fresh_frame(tmp_path):
    """Past a turn the re-anchored view (here a softer, different scene) no longer resembles chunk 0."""
    other = _drift(_scene(1), 1.0, 1.0)
    assert detail(other, detail_tiles(_scene())) < 0.65
    for name, img in (("ref", _scene()), ("shaded", _shaded(_scene())), ("other", other)):
        cv2.imwrite(str(tmp_path / f"{name}.png"), img)
    g = DriftGuard(min_gap=1, detail_below=0.65, reanchor_below=0.0)            # sharpness rule off: structure only
    g.reference({"front": tmp_path / "ref.png"})
    assert g({"front": tmp_path / "shaded.png"}, tmp_path / "c")[0] is None      # re-anchor
    out, st = g({"front": tmp_path / "other.png"}, tmp_path / "c")             # the re-anchored chunk's frame
    assert out is not None and st["front"]["detail"] == 1.0
    out, st = g({"front": tmp_path / "other.png"}, tmp_path / "c")
    assert out is not None and st["front"]["sat_ref"] == round(saturation(_scene()), 1)   # colour keeps chunk 0

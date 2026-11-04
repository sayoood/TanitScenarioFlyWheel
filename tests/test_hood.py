import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tsf.closedloop.neural import CosmosSensors
from tsf.cosmos.hood import BAND, LEAK_ABOVE, OUTLINE, Hood, hood_alpha

FRONT = "camera_front_wide_120fov"


def _inside(h: int, w: int) -> np.ndarray:
    """Pixels well inside the outline (its edge eroded by 8 px), above the band."""
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(OUTLINE * (w, h)).astype(np.int32)], 1)
    m = cv2.erode(m, np.ones((17, 17), np.uint8)).astype(bool)
    m[int(BAND * h) - 8:] = False
    return m


def test_a_redraw_without_a_hood_gets_chunk_0s_back(tmp_path):
    """The redraw at 33.6 s of the 40 s protected-left rerun drew no hood; the front view read like a side camera."""
    rng = np.random.default_rng(0)
    ref = rng.integers(90, 160, (720, 1280, 3), dtype=np.uint8)
    ref[_inside(720, 1280)] = (30, 25, 20)                                  # chunk 0's dark hood
    cv2.imwrite(str(tmp_path / "ref.png"), ref)
    road = rng.integers(90, 160, (480, 832, 3), dtype=np.uint8)              # a 480p redraw: road where the hood was
    cv2.imwrite(str(tmp_path / "last.png"), road)
    wr = cv2.VideoWriter(str(tmp_path / "v.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (832, 480))
    for _ in range(5):
        wr.write(road)
    wr.release()
    hood = Hood()
    hood.reference({FRONT: tmp_path / "ref.png", "camera_cross_left_120fov": tmp_path / "ref.png"})
    assert list(hood.ref) == [FRONT]
    hood({FRONT: str(tmp_path / "last.png")}, {FRONT: str(tmp_path / "v.mp4")})
    a = hood_alpha((480, 832))[..., 0]
    inside, away = _inside(480, 832), a < 1e-3
    last = cv2.imread(str(tmp_path / "last.png"))
    assert np.abs(last[inside].astype(int) - (30, 25, 20)).max() <= 2
    assert (last[away] == road[away]).all()
    cap, n = cv2.VideoCapture(str(tmp_path / "v.mp4")), 0
    while (fr := cap.read()[1]) is not None:
        assert fr[inside].mean() < 40 and fr[away].mean() > 100
        n += 1
    assert n == 5


def _frame(rim: int = 0, hood=(30, 25, 20), seed: int = 1) -> np.ndarray:
    """A 720p road with the hood (grown by ``rim`` px, as a redraw that draws its own larger hood) and a darker band."""
    img = np.random.default_rng(seed).integers(95, 106, (720, 1280, 3), dtype=np.uint8)
    m = np.zeros((720, 1280), np.uint8)
    cv2.fillPoly(m, [np.round(OUTLINE * (1280, 720)).astype(np.int32)], 1)
    if rim:
        m = cv2.dilate(m, np.ones((2 * rim + 1, 2 * rim + 1), np.uint8))
    img[m.astype(bool)] = hood
    img[int(BAND * 720):][m[int(BAND * 720):].astype(bool)] = (80, 80, 80)
    return img


def test_a_redraws_own_larger_hood_leaks_and_no_hood_does_not(tmp_path):
    """The 29.3 s redraw of the 720g rerun drew a light hood larger than chunk 0's; one with no hood at all is covered."""
    cv2.imwrite(str(tmp_path / "ref.png"), _frame())
    hood = Hood()
    hood.reference({FRONT: tmp_path / "ref.png"})
    assert hood.leak(_frame(seed=2), FRONT) < 0.05
    assert hood.leak(np.random.default_rng(3).integers(95, 106, (720, 1280, 3), dtype=np.uint8), FRONT) < LEAK_ABOVE
    assert hood.leak(_frame(rim=25, hood=(190, 190, 190)), FRONT) > LEAK_ABOVE
    assert hood.leak(cv2.resize(_frame(rim=25, hood=(190, 190, 190)), (832, 480)), FRONT) > LEAK_ABOVE


class _Leaky:
    """Hood stand-in: an attempt's last frame holds its leak score (x100) as its pixel value."""
    cams = (FRONT,)

    def leak(self, img, cam):
        return img[0, 0, 0] / 100


def _attempt(d, leak: float) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "last.png"), np.full((4, 4, 3), round(leak * 100), np.uint8))
    r = {"leak": leak, "last_frames": {FRONT: str(d / "last.png")}}
    (d / "result.json").write_text(json.dumps(r))
    return r


@pytest.fixture
def sensors(tmp_path):
    s = CosmosSensors.__new__(CosmosSensors)            # only what _hooded touches
    s.out, s.hood = tmp_path, _Leaky()
    return s


def test_a_leaking_redraw_is_rerolled_and_the_least_leaky_kept(sensors, tmp_path):
    gen, seeds = tmp_path / "gen", []
    rolls = iter([0.25, 0.1])
    r, info = sensors._hooded("t1_064", _attempt(gen / "t1_064", 0.3),
                              lambda d, off: seeds.append(off) or _attempt(d, next(rolls)))
    assert info == {"leak": [0.3, 0.25, 0.1], "kept": 2} and seeds == [2000, 3000]
    assert r["leak"] == 0.1 and Path(r["last_frames"][FRONT]).parent.name == "t1_064"
    assert json.loads((gen / "t1_064" / "result.json").read_text())["leak"] == 0.1
    assert (gen / "t1_064_rejected" / "result.json").exists()


def test_a_clean_redraw_or_worse_rerolls_keep_the_first(sensors, tmp_path):
    gen = tmp_path / "gen"
    first = _attempt(gen / "t1_024", 0.05)
    assert sensors._hooded("t1_024", first, lambda d, off: pytest.fail("no re-roll for a clean redraw")) == \
        (first, {"leak": [0.05], "kept": 0})
    rolls = iter([0.35, 0.4])
    first = _attempt(gen / "t1_045", 0.3)
    r, info = sensors._hooded("t1_045", first, lambda d, off: _attempt(d, next(rolls)))
    assert r is first and info == {"leak": [0.3, 0.35, 0.4], "kept": 0} and not (gen / "t1_045_rejected").exists()

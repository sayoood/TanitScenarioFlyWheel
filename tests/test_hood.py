import cv2
import numpy as np

from tsf.cosmos.hood import BAND, OUTLINE, Hood, hood_alpha

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

import json

import numpy as np
import pytest

from tsf.ingest.ego_speed import fuse_ego, idm_speed, window_starts


def test_window_starts_cover_the_span_and_stay_inside_the_video():
    assert window_starts(20.0) == [0.0, 6.0, 12.0, 18.0]
    assert window_starts(5.0) == [0.0]
    assert window_starts(20.0, total_s=21.0) == [0.0, 6.0, 12.0, pytest.approx(14.9)]
    assert window_starts(20.0, start_s=3.0) == [3.0, 9.0, 15.0, 21.0]


def test_fuse_ego_keeps_the_vipe_heading_and_integrates_the_new_speed():
    t = np.arange(0, 10.01, 0.1)
    yaw = np.where(t < 5, 0.0, np.pi / 2)                       # straight, then a left turn
    vipe_track = np.column_stack([np.zeros_like(t), np.zeros_like(t), yaw, np.zeros_like(t)])   # collapsed translation
    out = fuse_ego(vipe_track, t, np.array([0.0, 10.0]), np.array([20.0, 20.0]))
    assert np.allclose(out[:, 3], 20.0) and np.allclose(out[:, 2], yaw)
    assert out[-1, 0] == pytest.approx(100.0, abs=2.5) and out[-1, 1] == pytest.approx(100.0, abs=2.5)
    assert np.hypot(*np.diff(out[:, :2], axis=0).T).sum() == pytest.approx(200.0, abs=0.5)


def test_idm_speed_reads_cached_windows_and_scales_with_the_frozen_convention(tmp_path):
    from tsf.cosmos import idm
    act = np.zeros((60, 9))
    act[:, 2] = 1.0                                              # 1 m per 0.1 s step along the optical axis
    act[:, 3], act[:, 7] = 1.0, 1.0                              # identity rotation (6D columns)
    for s in (0, 6):
        (tmp_path / f"clip_{s:02d}.s0.json").write_text(json.dumps({"action": act.tolist()}))
    sp = idm_speed(tmp_path / "clip_0.mp4", 0.0, 10.0, tmp_path / "w", cache_dirs=(tmp_path,))
    assert sp["windows"] == 2 and sp["cached"] == 2 and not sp["latency_s"]
    assert sp["t"][0] == 0.0 and sp["t"][-1] == pytest.approx(12.0)
    assert np.allclose(sp["v"][5:55], 10.0 * idm.CONVENTION["scale"], atol=0.2)
    with pytest.raises(RuntimeError, match="no cached"):
        idm_speed(tmp_path / "clip_0.mp4", 0.0, 20.0, tmp_path / "w2", cache_dirs=(tmp_path,),
                  base_url="http://127.0.0.1:9")

import json
from pathlib import Path

import pytest

from tsf.closedloop.neural import CosmosSensors


def _attempt(d: Path, tag: str) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    r = {"tag": tag, "videos": {"cam": str(d / "cam.mp4")}, "last_frames": {"cam": str(d / "last_cam.png")}}
    (d / "result.json").write_text(json.dumps(r))
    return r


@pytest.fixture
def sensors(tmp_path):
    s = CosmosSensors.__new__(CosmosSensors)            # only what _judged touches
    s.out = tmp_path
    return s


def test_a_better_reroll_takes_the_chunk_directory(sensors, tmp_path):
    sensors._verdicts = lambda cid, r: {"first": ["wrong", "match"], "retry": ["match"]}[r["tag"]]
    r, info = sensors._judged("t1_003", _attempt(tmp_path / "gen" / "t1_003", "first"),
                              lambda d, off: _attempt(d, "retry"))
    assert info == {"first": ["wrong", "match"], "retry": ["match"], "kept": "retry"}
    assert r["tag"] == "retry" and "_retry" not in json.dumps(r) and Path(r["last_frames"]["cam"]).parent.name == "t1_003"
    assert json.loads((tmp_path / "gen" / "t1_003" / "result.json").read_text())["videos"]["cam"].endswith("t1_003/cam.mp4")
    assert (tmp_path / "gen" / "t1_003_rejected" / "result.json").exists()


def test_a_clean_or_not_better_attempt_is_kept(sensors, tmp_path):
    sensors._verdicts = lambda cid, r: {"first": ["match"], "retry": ["wrong"]}[r["tag"]]
    first = _attempt(tmp_path / "gen" / "t1_004", "first")
    assert sensors._judged("t1_004", first, lambda d, off: pytest.fail("no re-roll for a clean chunk")) == \
        (first, {"first": ["match"]})
    sensors._verdicts = lambda cid, r: {"first": ["wrong"], "retry": ["wrong"]}[r["tag"]]
    r, info = sensors._judged("t1_005", _attempt(tmp_path / "gen" / "t1_005", "first"), lambda d, off: _attempt(d, "retry"))
    assert r["tag"] == "first" and info["kept"] == "first" and not (tmp_path / "gen" / "t1_005_rejected").exists()

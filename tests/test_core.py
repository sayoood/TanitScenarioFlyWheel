import json
import math
from pathlib import Path

import numpy as np
import pytest

from tsf.ir.scenario import Scenario
from tsf.sim.odr import Geometry
from tsf.verify import ladder

EX = Path(__file__).resolve().parents[1] / "examples"


def load(name: str) -> Scenario:
    return Scenario.model_validate_json((EX / name).read_text())


def test_spiral_curvature_is_linear():
    g = Geometry(0, 1.0, 2.0, 0.3, 80.0, "spiral", {"curvStart": 0.0, "curvEnd": 0.02})
    s = np.linspace(0, 80, 801)
    x, y, h = g.eval(s)
    assert (x[0], y[0], h[0]) == pytest.approx((1.0, 2.0, 0.3), abs=1e-9)
    k = np.gradient(h, s)
    assert k[10:-10] == pytest.approx(0.02 * s[10:-10] / 80, abs=2e-4)
    step = np.hypot(np.diff(x), np.diff(y))
    assert step == pytest.approx(0.1, abs=1e-4)  # arc-length parametrised


def test_arc_and_line_chain():
    g = Geometry(0, 0, 0, 0, math.pi * 10, "arc", {"curvature": 0.1})
    x, y, h = g.eval(np.array([math.pi * 10]))
    assert (x[0], y[0], h[0]) == pytest.approx((0.0, 20.0, math.pi), abs=1e-6)


@pytest.mark.parametrize("name", ["uc1_left_turn_intersection.json", "double_cut_in_highway.json"])
def test_examples_pass_ladder(name, tmp_path):
    rep = ladder.run(load(name), tmp_path, qc=False)
    print(json.dumps(rep.to_dict()["findings"], indent=1), rep.metrics)
    assert rep.passed, rep.diagnostics()

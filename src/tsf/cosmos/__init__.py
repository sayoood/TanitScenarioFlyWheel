"""Bridge from TanitSim to NVIDIA Cosmos 3 World Scenario Map (WSM v4) control videos.

    TanitSim run (OdrMap + SimLog)
      -> rds_hq.export_clip   RDS-HQ clip (one tar per component, 30 Hz, Hyperion rig)
      -> wsm.render_wsm       per-camera WSM v4 control MP4s (NVIDIA ``cosmos-auto-wm`` renderer)
      -> wsm.mosaic           labelled 7-camera grid video

The renderer and NVIDIA's RDS-HQ writer/validator are *not* vendored: they are read from the
Cosmos framework checkout (``TSF_WSM_DIR``, default
``~/tsf/third_party/cosmos-framework/.agents/skills/cookbooks/cosmos3/simulation/auto/wsm``) and
run in a separate venv (``TSF_WSM_PYTHON``, default ``~/tsf/venvs/wsm/bin/python``).

Importing this package is cheap; submodules load lazily.
"""
from __future__ import annotations

import importlib

__all__ = ["export_clip", "choose_t0", "render_wsm", "mosaic", "run_example"]

_LAZY = {
    "export_clip": "rds_hq", "choose_t0": "rds_hq",
    "render_wsm": "wsm", "mosaic": "wsm",
    "run_example": "pipeline",
}


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(importlib.import_module(f"{__name__}.{_LAZY[name]}"), name)
    raise AttributeError(name)

"""``python -m tsf.agents.augment`` against a Sim-Auto server that is already running and belongs to this job.

``augment_render.render`` refuses to use a Cosmos server it did not start itself (it waits for the slot to be free, so that
it never takes over another job's server). Here the running server was started by the caller (``pipeline.ensure_server``,
owner label ``tsf.owner``), so that guard is switched off and ``ensure_server`` reuses it; the server is left running.

    python scripts/augment_on_running_server.py examples/uc1_left_turn_intersection.json runs/augment_demo/auto_seed0_render --mode auto --seed 0
"""
from __future__ import annotations

import sys

from tsf.agents import augment, augment_render

if __name__ == "__main__":
    from tsf.cosmos import generate
    running = generate.running_servers()
    if running != ["tsf-cosmos-sim"]:
        sys.exit(f"expected exactly our tsf-cosmos-sim to be running, found {running}")
    print(f"using the running Cosmos server {running[0]} (owner {generate.server_owner(running[0])!r})", flush=True)
    augment_render._running_cosmos = lambda: []          # noqa: SLF001 - the running server is ours
    augment.main(sys.argv[1:])

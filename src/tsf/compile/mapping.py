"""Single definition of TSIR LanePosition -> OpenDRIVE (road id, lane id, road s). See xodr.py conventions."""
from __future__ import annotations

from tsf.ir.scenario import Junction, LanePosition, LinearRoad, Scenario


def road_length(sc: Scenario, road_id: int) -> float:
    if isinstance(sc.road, LinearRoad):
        return sum(seg.length for seg in sc.road.segments)
    return sc.road.approaches[road_id].length


def odr_position(sc: Scenario, pos: LanePosition) -> tuple[str, int, float]:
    if isinstance(sc.road, LinearRoad):
        if pos.direction in ("forward", "outbound"):
            return "0", -pos.lane, pos.s
        return "0", pos.lane, road_length(sc, 0) - pos.s
    if isinstance(sc.road, Junction):
        lid = pos.lane if pos.direction in ("inbound", "backward") else -pos.lane
        return str(pos.road), lid, pos.s
    raise NotImplementedError("positions on imported OpenDRIVE maps use road/lane ids directly")

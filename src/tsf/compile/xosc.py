"""TSIR -> ASAM OpenSCENARIO XML 1.3 (.xosc) via scenariogeneration.

One ManeuverGroup per actor, one Event per TSIR action; TSIR triggers map 1:1 onto OSC
conditions, `after_action` onto StoryboardElementStateCondition (startTransition/endTransition).
Signal plans become a TrafficSignalController in the RoadNetwork.
"""
from __future__ import annotations

from pathlib import Path

from scenariogeneration import xosc

from tsf.compile.mapping import odr_position
from tsf.compile.xodr import SIGNAL_ID_BASE
from tsf.ir.scenario import Junction, Scenario
from tsf.sim.engine import SignalPlan

_CAT = {"car": "car", "suv": "car", "van": "van", "truck": "truck", "bus": "bus", "motorcycle": "motorbike",
        "bicycle": "bicycle", "pedestrian": "pedestrian"}


def _entity(a):
    l, w, h = a.dims()
    bb = xosc.BoundingBox(w, l, h, l * 0.25, 0.0, h / 2)
    if a.category == "pedestrian":
        return xosc.Pedestrian(a.id, 80, xosc.PedestrianCategory.pedestrian, bb, model="pedestrian")
    wb = min(3.0, l * 0.6)
    fa = xosc.Axle(0.5, 0.7, w * 0.85, wb, 0.35)
    ra = xosc.Axle(0.0, 0.7, w * 0.85, 0.0, 0.35)
    veh = xosc.Vehicle(a.id, getattr(xosc.VehicleCategory, _CAT[a.category]), bb, fa, ra,
                       max_speed=70, max_acceleration=10, max_deceleration=10)
    veh.add_property("color", a.color)
    veh.add_property("tsf_category", a.category)
    return veh


def _speed(v, rate):
    return xosc.AbsoluteSpeedAction(v, xosc.TransitionDynamics(xosc.DynamicsShapes.linear,
                                                               xosc.DynamicsDimension.rate, rate))


def _trigger(sc: Scenario, trig, name: str):
    k = trig.kind
    edge = xosc.ConditionEdge.rising
    if k == "sim_time":
        return xosc.ValueTrigger(name, 0, xosc.ConditionEdge.none,
                                 xosc.SimulationTimeCondition(trig.t, xosc.Rule.greaterThan))
    if k == "after_action":
        state = xosc.StoryboardElementState.startTransition if trig.on == "start" \
            else xosc.StoryboardElementState.endTransition
        return xosc.ValueTrigger(name, trig.delay, edge,
                                 xosc.StoryboardElementStateCondition(xosc.StoryboardElementType.action,
                                                                      trig.action_id, state))
    if k == "distance_to_actor":
        rule = xosc.Rule.lessThan if trig.rule == "less" else xosc.Rule.greaterThan
        if trig.mode == "euclidean":
            cond = xosc.RelativeDistanceCondition(trig.distance, rule, xosc.RelativeDistanceType.euclidianDistance,
                                                  trig.other, freespace=trig.freespace)
        else:  # signed: OSC 1.3 RelativeClearanceCondition-free form = longitudinal distance in other's frame
            cond = xosc.RelativeDistanceCondition(trig.distance, rule, xosc.RelativeDistanceType.longitudinal,
                                                  trig.other, freespace=trig.freespace,
                                                  coordinate_system=xosc.CoordinateSystem.entity)
        return xosc.EntityTrigger(name, 0, edge, cond, trig.actor)
    if k == "ttc":
        cond = xosc.TimeToCollisionCondition(trig.ttc, xosc.Rule.lessThan, entity=trig.other, freespace=True)
        return xosc.EntityTrigger(name, 0, edge, cond, trig.actor)
    if k == "distance_to_stop_line":
        a = next(x for x in sc.actors if x.id == trig.actor)
        rid, lid, _ = odr_position(sc, a.initial)
        cond = xosc.DistanceCondition(trig.distance, xosc.Rule.lessThan, xosc.LanePosition(0.0, 0, lid, rid),
                                      alongroute=False, freespace=False)
        return xosc.EntityTrigger(name, 0, edge, cond, trig.actor)
    raise ValueError(k)


def _action(sc: Scenario, act):
    t = act.type
    if t == "speed_change":
        return _speed(act.target_speed, act.accel)
    if t == "lane_change":
        n = act.lanes if act.direction == "left" else -act.lanes
        dyn = xosc.TransitionDynamics(xosc.DynamicsShapes.sinusoidal, xosc.DynamicsDimension.time, act.duration)
        return xosc.RelativeLaneChangeAction(n, act.actor, dyn)
    if t == "cut_in":
        dyn = xosc.TransitionDynamics(xosc.DynamicsShapes.sinusoidal, xosc.DynamicsDimension.time, act.duration)
        return xosc.RelativeLaneChangeAction(0, act.target, dyn)
    if t == "follow":
        return xosc.LongitudinalTimegapAction(act.target, act.time_gap, continuous=True, freespace=True)
    if t == "stop_at_line":
        return _speed(0.0, act.decel)
    if t == "brake":
        return _speed(0.0, act.decel)
    if t == "cross":
        return _speed(act.speed, 2.0)
    raise ValueError(t)


def _signal_controller(sc: Scenario, road_network: xosc.RoadNetwork):
    plan = SignalPlan(sc)
    if not plan.enabled:
        return
    n = len(sc.road.approaches)
    ctrl = xosc.TrafficSignalController("tsf_junction_ctrl", delay=plan.offset)
    for pi, (arms, turns, green) in enumerate(plan.phases):
        for label, dur, lamp in (("green", green, "off;off;on"), ("yellow", plan.yellow, "off;on;off"),
                                 ("allred", plan.red, "on;off;off")):
            ph = xosc.Phase(f"p{pi}_{label}", dur)
            for arm in range(n):
                ph.add_signal_state(str(SIGNAL_ID_BASE + arm), lamp if arm in arms else "on;off;off")
            ctrl.add_phase(ph)
    road_network.add_traffic_signal_controller(ctrl)


def build(sc: Scenario, xodr_name: str) -> xosc.Scenario:
    road = xosc.RoadNetwork(roadfile=xodr_name)
    if isinstance(sc.road, Junction):
        _signal_controller(sc, road)
    entities = xosc.Entities()
    init = xosc.Init()
    for a in sc.actors:
        entities.add_scenario_object(a.id, _entity(a))
        rid, lid, s = odr_position(sc, a.initial)
        init.add_init_action(a.id, xosc.TeleportAction(xosc.LanePosition(s, a.initial.t_offset, lid, rid)))
        init.add_init_action(a.id, xosc.AbsoluteSpeedAction(
            a.initial_speed, xosc.TransitionDynamics(xosc.DynamicsShapes.step, xosc.DynamicsDimension.time, 0)))
    act = xosc.Act("tsf_act", xosc.ValueTrigger("act_start", 0, xosc.ConditionEdge.none,
                                                xosc.SimulationTimeCondition(0, xosc.Rule.greaterThan)))
    for a in sc.actors:
        mine = [x for x in sc.actions if x.actor == a.id]
        if not mine:
            continue
        man = xosc.Maneuver(f"{a.id}_maneuver")
        for x in mine:
            ev = xosc.Event(f"{x.id}_event", xosc.Priority.parallel)
            ev.add_action(x.id, _action(sc, x))
            ev.add_trigger(_trigger(sc, x.start, f"{x.id}_start"))
            man.add_event(ev)
        mg = xosc.ManeuverGroup(f"{a.id}_group")
        mg.add_actor(a.id)
        mg.add_maneuver(man)
        act.add_maneuver_group(mg)
    story = xosc.Story("tsf_story")
    story.add_act(act)
    stop = xosc.ValueTrigger("stop", 0, xosc.ConditionEdge.rising,
                             xosc.SimulationTimeCondition(sc.duration, xosc.Rule.greaterThan), "stop")
    sb = xosc.StoryBoard(init, stop)
    sb.add_story(story)
    return xosc.Scenario(sc.meta.title[:120], "Tanit Scenario Flywheel", xosc.ParameterDeclarations(), entities,
                         sb, road, xosc.Catalog(), osc_minor_version=3)


def write(sc: Scenario, path: str | Path, xodr_name: str | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    build(sc, xodr_name or f"{sc.meta.id}.xodr").write_xml(str(path))
    return path

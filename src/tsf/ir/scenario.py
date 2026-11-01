"""Tanit Scenario IR (TSIR): the single typed contract between all flywheel stages.

Every source (text, dashcam video, incident report, sensor snippet) is lifted into a
`Scenario`; every sink (OpenDRIVE, OpenSCENARIO 1.x XML, OSC 2.x DSL, TanitSim, WSM
renderer, Cosmos conditioning, closed loop) is compiled from it. The LLM never writes
XML directly: it emits TSIR JSON under schema-constrained decoding, and deterministic
compilers do the rest. Units are SI (m, s, m/s, rad) unless a field name says otherwise.
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

TSIR_VERSION = "0.1.0"


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


# --------------------------------------------------------------------------- environment
class Weather(str, Enum):
    clear = "clear"
    cloudy = "cloudy"
    rain = "rain"
    heavy_rain = "heavy_rain"
    fog = "fog"
    snow = "snow"


class RoadSurface(str, Enum):
    dry = "dry"
    wet = "wet"
    snow = "snow"
    ice = "ice"


class Environment(_M):
    time_of_day: str = Field("12:00", pattern=r"^([01]\d|2[0-3]):[0-5]\d$", description="local time HH:MM")
    weather: Weather = Weather.clear
    precipitation: float = Field(0.0, ge=0, le=1, description="0 none .. 1 extreme")
    fog_visibility_m: float = Field(100000.0, gt=0)
    cloud_cover: float = Field(0.1, ge=0, le=1)
    road_surface: RoadSurface = RoadSurface.dry
    sun_azimuth_deg: float | None = Field(None, description="override; else derived from time_of_day")


# --------------------------------------------------------------------------- road network
class LaneConfig(_M):
    driving_lanes: int = Field(2, ge=1, le=6, description="through lanes in this direction")
    lane_width: float = Field(3.5, ge=2.5, le=4.5)
    left_turn_lane: bool = False
    right_turn_lane: bool = False
    turn_lane_length: float = Field(40.0, ge=10, le=200)


class Approach(_M):
    """One arm of a junction (or one road end), heading away from the junction centre."""
    heading_deg: float = Field(..., description="arm direction, 0 = +x (east), CCW positive")
    length: float = Field(150.0, ge=20, le=2000)
    inbound: LaneConfig = Field(default_factory=LaneConfig, description="lanes driving toward the junction")
    outbound: LaneConfig = Field(default_factory=LaneConfig, description="lanes driving away")
    speed_limit_kph: float = Field(50, gt=0, le=200)


class TrafficLightPlan(_M):
    green_s: float = Field(20.0, gt=0)
    yellow_s: float = Field(3.0, ge=0)
    all_red_s: float = Field(2.0, ge=0)
    protected_left: bool = False
    initial_phase_offset_s: float = 0.0


class RoadSegment(_M):
    """Straight / constant-curvature / clothoid chain, used for highways and rural roads."""
    kind: Literal["line", "arc", "spiral"] = "line"
    length: float = Field(..., gt=0)
    curvature_start: float = 0.0
    curvature_end: float = 0.0


class LinearRoad(_M):
    type: Literal["linear"] = "linear"
    segments: list[RoadSegment] = Field(default_factory=lambda: [RoadSegment(length=1000)])
    forward: LaneConfig = Field(default_factory=LaneConfig)
    backward: LaneConfig | None = Field(default_factory=LaneConfig, description="None = one-way road")
    speed_limit_kph: float = 130
    median_width: float = Field(0.0, ge=0, description=">0 models a divided highway")
    shoulder_width: float = Field(2.5, ge=0)


class Junction(_M):
    type: Literal["junction"] = "junction"
    approaches: list[Approach] = Field(..., min_length=3, max_length=5)
    signalized: bool = True
    signal_plan: TrafficLightPlan = Field(default_factory=TrafficLightPlan)
    corner_radius: float = Field(12.0, ge=3, le=60)


class FromOpenDrive(_M):
    """Use an existing map (e.g. reconstructed from video or an incident location)."""
    type: Literal["opendrive_file"] = "opendrive_file"
    path: str


RoadNetwork = Annotated[Union[LinearRoad, Junction, FromOpenDrive], Field(discriminator="type")]


# --------------------------------------------------------------------------- actors
class ActorCategory(str, Enum):
    car = "car"
    suv = "suv"
    van = "van"
    truck = "truck"
    bus = "bus"
    motorcycle = "motorcycle"
    bicycle = "bicycle"
    pedestrian = "pedestrian"


DEFAULT_DIMS: dict[str, tuple[float, float, float]] = {  # length, width, height
    "car": (4.6, 1.85, 1.5), "suv": (4.9, 1.95, 1.75), "van": (5.3, 2.05, 2.2),
    "truck": (12.0, 2.55, 3.6), "bus": (12.5, 2.55, 3.2), "motorcycle": (2.2, 0.8, 1.4),
    "bicycle": (1.8, 0.6, 1.7), "pedestrian": (0.5, 0.6, 1.75),
}


class LanePosition(_M):
    """Position on the road graph. `road` is an approach index (junction) or 0 (linear road).
    `lane` counts from the centre line: 1 = innermost lane of the given direction."""
    road: int = 0
    direction: Literal["forward", "backward", "inbound", "outbound"] = "forward"
    lane: int = Field(1, ge=1)
    s: float = Field(..., description="m along the road in driving direction; junction arms: distance to stop line")
    t_offset: float = Field(0.0, description="lateral offset from lane centre, + = left")


class Actor(_M):
    id: str = Field(..., pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    role: Literal["ego", "npc"] = "npc"
    category: ActorCategory = ActorCategory.car
    dimensions: tuple[float, float, float] | None = Field(None, description="l, w, h; default per category")
    color: str = "silver"
    initial: LanePosition
    initial_speed: float = Field(0.0, ge=0)
    route: list[Literal["straight", "left", "right", "u_turn"]] = Field(
        default_factory=list, description="turn choice at each junction encountered, in order")
    headway: float | None = Field(None, ge=0.4, le=3.0, description="driver model: desired time headway to the "
                                  "vehicle ahead in s (default 1.5); lifted from logs for replayed drivers")

    def dims(self) -> tuple[float, float, float]:
        return self.dimensions or DEFAULT_DIMS[self.category]


# --------------------------------------------------------------------------- triggers & actions
class SimTime(_M):
    kind: Literal["sim_time"] = "sim_time"
    t: float = Field(..., ge=0)


class DistanceToActor(_M):
    """euclidean: |actor-other|; ahead/behind: signed bumper gap of `actor` ahead of / behind `other`
    along other's heading (fires only on that side)."""
    kind: Literal["distance_to_actor"] = "distance_to_actor"
    actor: str
    other: str
    distance: float = Field(..., ge=0)
    rule: Literal["less", "greater"] = "less"
    freespace: bool = True
    mode: Literal["euclidean", "ahead", "behind"] = "euclidean"


class TimeToCollision(_M):
    kind: Literal["ttc"] = "ttc"
    actor: str
    other: str
    ttc: float = Field(..., gt=0)


class DistanceToStopLine(_M):
    kind: Literal["distance_to_stop_line"] = "distance_to_stop_line"
    actor: str
    distance: float = Field(..., ge=0)


class AfterAction(_M):
    """OSC1 StoryboardElementStateCondition: fires `delay` s after the referenced action starts/ends."""
    kind: Literal["after_action"] = "after_action"
    action_id: str
    on: Literal["start", "end"] = "end"
    delay: float = Field(0.0, ge=0)


Trigger = Annotated[
    Union[SimTime, DistanceToActor, TimeToCollision, DistanceToStopLine, AfterAction],
    Field(discriminator="kind")]


class _Action(_M):
    id: str
    actor: str
    start: Trigger = Field(default_factory=lambda: SimTime(t=0.0))


class SpeedChange(_Action):
    type: Literal["speed_change"] = "speed_change"
    target_speed: float = Field(..., ge=0)
    accel: float = Field(2.0, gt=0, le=12, description="|a| used for the change, m/s^2")


class LaneChange(_Action):
    type: Literal["lane_change"] = "lane_change"
    direction: Literal["left", "right"]
    lanes: int = Field(1, ge=1, le=3)
    duration: float = Field(4.0, gt=0.5, le=15)
    safe_gap: bool = Field(False, description="replayed drivers: wait up to 5 s for a safe gap in the target lane "
                           "(MOBIL safety criterion), else skip the lane change; scripted timing when false")


class FollowActor(_Action):
    type: Literal["follow"] = "follow"
    target: str
    time_gap: float = Field(1.8, gt=0.3, le=5)


class CutIn(_Action):
    type: Literal["cut_in"] = "cut_in"
    target: str = Field(..., description="actor whose lane is entered (usually ego)")
    gap: float = Field(12.0, gt=0, description="longitudinal gap to target at cut-in end, m")
    duration: float = Field(3.0, gt=0.5, le=10)


class StopAtLine(_Action):
    type: Literal["stop_at_line"] = "stop_at_line"
    decel: float = Field(2.5, gt=0, le=10)


class EmergencyBrake(_Action):
    type: Literal["brake"] = "brake"
    decel: float = Field(6.0, gt=0, le=12)


class CrossRoad(_Action):
    """Pedestrian / cyclist crosses the carriageway (crosswalk if present)."""
    type: Literal["cross"] = "cross"
    speed: float = Field(1.4, gt=0, le=8)
    side: Literal["left_to_right", "right_to_left"] = "right_to_left"


class LaneOffset(_Action):
    """In-lane lateral shift (e.g. nudging past a vehicle that partly blocks the lane); kept until changed."""
    type: Literal["lane_offset"] = "lane_offset"
    offset: float = Field(..., ge=-2.5, le=2.5, description="target offset from the lane centre, m, + = left")
    duration: float = Field(2.0, gt=0.3, le=10)


Action = Annotated[
    Union[SpeedChange, LaneChange, FollowActor, CutIn, StopAtLine, EmergencyBrake, CrossRoad, LaneOffset],
    Field(discriminator="type")]


# --------------------------------------------------------------------------- oracle
class Assertion(_M):
    """Machine-checkable intent: the verifier stack replays the scenario and checks these."""
    kind: Literal[
        "no_collision", "min_ttc_above", "ego_lane_change", "ego_turn", "ego_stops_before_line",
        "ego_speed_between", "ego_reaches_s", "red_light_respected", "max_decel_below",
    ]
    actor: str = "ego"
    value: float | None = Field(None, description="min_ttc_above: s; ego_lane_change: net lane changes to the left "
                                "(negative = right, default 1); max_decel_below: m/s2; ego_speed_between: lower bound "
                                "m/s; ego_reaches_s: distance in m")
    value2: float | None = Field(None, description="ego_speed_between: upper bound m/s")
    label: str | None = Field(None, description="ego_turn: the turn direction, exactly 'left', 'right' or "
                              "'straight'; otherwise unused")


# --------------------------------------------------------------------------- logical parameters
class ParameterRange(_M):
    """Logical-scenario parameter for augmentation/search. `path` is a JSON pointer into the IR."""
    name: str
    path: str = Field(..., pattern=r"^/")
    distribution: Literal["uniform", "normal", "choice"] = "uniform"
    low: float | None = None
    high: float | None = None
    choices: list[Union[str, float]] | None = None


class SensorRigRef(_M):
    name: Literal["nvidia_hyperion_av", "custom"] = "nvidia_hyperion_av"
    path: str | None = None


class Meta(_M):
    id: str
    title: str
    description: str = ""
    source: Literal["text", "dashcam", "incident", "sensor_snippet", "negative_snippet", "agent"] = "text"
    polarity: Literal["positive", "negative"] = "positive"
    tags: list[str] = Field(default_factory=list)
    parent_id: str | None = Field(None, description="scenario this one was derived from (twin / variant)")


class Scenario(_M):
    tsir_version: str = TSIR_VERSION
    meta: Meta
    environment: Environment = Field(default_factory=Environment)
    road: RoadNetwork
    actors: list[Actor] = Field(..., min_length=1)
    actions: list[Action] = Field(default_factory=list)
    assertions: list[Assertion] = Field(default_factory=list)
    parameters: list[ParameterRange] = Field(default_factory=list)
    duration: float = Field(20.0, gt=0, le=600)
    dt: float = Field(0.05, gt=0, le=0.2)
    sensor_rig: SensorRigRef = Field(default_factory=SensorRigRef)

    @model_validator(mode="after")
    def _referential_integrity(self) -> "Scenario":
        ids = [a.id for a in self.actors]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate actor ids")
        if sum(a.role == "ego" for a in self.actors) != 1:
            raise ValueError("exactly one actor must have role 'ego'")
        known, action_ids = set(ids), set()
        for act in self.actions:
            if act.id in action_ids:
                raise ValueError(f"duplicate action id {act.id}")
            action_ids.add(act.id)
            refs = [act.actor] + [getattr(act, f) for f in ("target",) if hasattr(act, f)]
            trig = act.start
            refs += [getattr(trig, f) for f in ("actor", "other") if hasattr(trig, f)]
            for r in refs:
                if r not in known:
                    raise ValueError(f"action {act.id} references unknown actor '{r}'")
        for act in self.actions:
            if getattr(act.start, "kind", None) == "after_action" and act.start.action_id not in action_ids:
                raise ValueError(f"action {act.id} waits for unknown action '{act.start.action_id}'")
        if isinstance(self.road, Junction):
            n = len(self.road.approaches)
            for a in self.actors:
                if a.initial.road >= n:
                    raise ValueError(f"actor {a.id} on approach {a.initial.road}, junction has {n}")
        return self

    @property
    def ego(self) -> Actor:
        return next(a for a in self.actors if a.role == "ego")


def json_schema() -> dict:
    """Schema handed to vLLM structured outputs (xgrammar) for constrained decoding."""
    return Scenario.model_json_schema()

/** Mock artifact text: OpenDRIVE, OpenSCENARIO 1.3, OpenSCENARIO DSL 2.x and the IMDAD package. */
import { ARM_HEADINGS, ARM_LEN, BOX, LANE_W, getConnectors } from "./scene";

const f = (v: number, d = 4) => (Math.round(v * 10 ** d) / 10 ** d).toString();
const rad = (deg: number) => (deg * Math.PI) / 180;

export function makeXodr(name: string): string {
  const out: string[] = [];
  out.push('<?xml version="1.0" encoding="UTF-8"?>');
  out.push("<OpenDRIVE>");
  out.push(`  <header revMajor="1" revMinor="8" name="${name}" version="1.0" date="2026-11-01T11:30:00" vendor="Tanit Scenario Flywheel (tsf.compile.xodr)">`);
  out.push(`    <geoReference><![CDATA[+proj=tmerc +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs]]></geoReference>`);
  out.push("  </header>");
  // arm roads: reference line starts at the junction and runs outward; left lanes are inbound
  for (let a = 0; a < 4; a++) {
    const len = ARM_LEN[a] - BOX;
    const x0 = Math.cos(rad(ARM_HEADINGS[a])) * BOX;
    const y0 = Math.sin(rad(ARM_HEADINGS[a])) * BOX;
    out.push(`  <road name="arm_${a}" length="${f(len)}" id="${a}" junction="-1">`);
    out.push("    <link>");
    out.push(`      <predecessor elementType="junction" elementId="900"/>`);
    out.push("    </link>");
    out.push("    <type s=\"0\" type=\"town\"><speed max=\"50\" unit=\"km/h\"/></type>");
    out.push("    <planView>");
    out.push(`      <geometry s="0" x="${f(x0)}" y="${f(y0)}" hdg="${f(rad(ARM_HEADINGS[a]))}" length="${f(len)}"><line/></geometry>`);
    out.push("    </planView>");
    out.push("    <lanes>");
    out.push('      <laneSection s="0">');
    out.push("        <left>");
    for (const id of [2, 1]) {
      out.push(`          <lane id="${id}" type="driving" level="false">`);
      out.push(`            <width sOffset="0" a="${LANE_W}" b="0" c="0" d="0"/>`);
      out.push(`            <roadMark sOffset="0" type="${id === 2 ? "solid" : "broken"}" weight="standard" color="white" width="0.12"/>`);
      out.push("          </lane>");
    }
    out.push("        </left>");
    out.push("        <center>");
    out.push('          <lane id="0" type="none" level="false"><roadMark sOffset="0" type="solid solid" color="yellow" width="0.12"/></lane>');
    out.push("        </center>");
    out.push("        <right>");
    for (const id of [-1, -2]) {
      out.push(`          <lane id="${id}" type="driving" level="false">`);
      out.push(`            <width sOffset="0" a="${LANE_W}" b="0" c="0" d="0"/>`);
      out.push(`            <roadMark sOffset="0" type="${id === -2 ? "solid" : "broken"}" weight="standard" color="white" width="0.12"/>`);
      out.push("          </lane>");
    }
    out.push("        </right>");
    out.push("      </laneSection>");
    out.push("    </lanes>");
    out.push("    <signals>");
    out.push(`      <signal s="${f(len - 1)}" t="${f(-2 * LANE_W - 1.4)}" id="${200 + a}" name="tl_arm${a}" dynamic="yes" orientation="-" zOffset="4.2" country="DE" type="1000001" subtype="-1" hOffset="0" width="0.4" height="0.9"/>`);
    out.push("    </signals>");
    out.push("  </road>");
  }
  // connecting roads (one per movement, cubic Bezier expressed as paramPoly3)
  const conns = getConnectors();
  conns.forEach((c, i) => {
    const [E, P1, P2, X] = c.ctrl;
    const h = c.entryHeading;
    const loc = (p: [number, number]): [number, number] => {
      const dx = p[0] - E[0];
      const dy = p[1] - E[1];
      return [dx * Math.cos(-h) - dy * Math.sin(-h), dx * Math.sin(-h) + dy * Math.cos(-h)];
    };
    const q0 = loc(E);
    const q1 = loc(P1);
    const q2 = loc(P2);
    const q3 = loc(X);
    const poly = (k: 0 | 1) => {
      const a0 = q0[k];
      const b0 = 3 * (q1[k] - q0[k]);
      const c0 = 3 * (q0[k] - 2 * q1[k] + q2[k]);
      const d0 = -q0[k] + 3 * q1[k] - 3 * q2[k] + q3[k];
      return [a0, b0, c0, d0].map((v) => f(v, 5));
    };
    const [aU, bU, cU, dU] = poly(0);
    const [aV, bV, cV, dV] = poly(1);
    const id = 100 + i;
    out.push(`  <road name="conn_${c.id}" length="${f(c.length)}" id="${id}" junction="900">`);
    out.push("    <link>");
    out.push(`      <predecessor elementType="road" elementId="${c.arm}" contactPoint="start"/>`);
    out.push(`      <successor elementType="road" elementId="${c.outArm}" contactPoint="start"/>`);
    out.push("    </link>");
    out.push("    <planView>");
    out.push(`      <geometry s="0" x="${f(E[0])}" y="${f(E[1])}" hdg="${f(h)}" length="${f(c.length)}">`);
    out.push(`        <paramPoly3 aU="${aU}" bU="${bU}" cU="${cU}" dU="${dU}" aV="${aV}" bV="${bV}" cV="${cV}" dV="${dV}" pRange="normalized"/>`);
    out.push("      </geometry>");
    out.push("    </planView>");
    out.push('    <lanes><laneSection s="0"><right>');
    out.push(`      <lane id="-1" type="driving" level="false"><width sOffset="0" a="${LANE_W}" b="0" c="0" d="0"/></lane>`);
    out.push("    </right></laneSection></lanes>");
    out.push("  </road>");
  });
  out.push('  <junction name="main_junction" id="900" type="default">');
  conns.forEach((c, i) => {
    out.push(`    <connection id="${i}" incomingRoad="${c.arm}" connectingRoad="${100 + i}" contactPoint="start">`);
    out.push(`      <laneLink from="${-c.lane}" to="-1"/>`);
    out.push("    </connection>");
  });
  out.push("  </junction>");
  out.push("</OpenDRIVE>");
  return out.join("\n") + "\n";
}

export function makeXosc(name: string, query: string): string {
  const actors: [string, string, string, string, number][] = [
    ["ego", "car", "white", "2", 12.5],
    ["lead_right", "van", "blue", "2", 12],
    ["follow_left", "car", "black", "2", 11],
    ["oncoming_1", "car", "red", "0", 13],
    ["oncoming_2", "truck", "white", "0", 11.5],
    ["cross_north", "car", "grey", "1", 8],
    ["cross_south", "bus", "yellow", "3", 8],
  ];
  const dims: Record<string, [number, number, number]> = { car: [4.6, 1.9, 1.5], van: [5.4, 2, 2.1], truck: [8.5, 2.5, 3.3], bus: [12, 2.55, 3.2] };
  const startS: Record<string, number> = { ego: 140, lead_right: 115, follow_left: 175, oncoming_1: 120, oncoming_2: 165, cross_north: 60, cross_south: 70 };
  const lanes: Record<string, number> = { ego: 2, lead_right: 2, follow_left: 1, oncoming_1: 2, oncoming_2: 2, cross_north: 1, cross_south: 2 };
  const o: string[] = [];
  o.push('<?xml version="1.0" encoding="UTF-8"?>');
  o.push("<OpenSCENARIO>");
  o.push(`  <FileHeader revMajor="1" revMinor="3" date="2026-11-01T11:30:00" description="${query.slice(0, 96).replace(/"/g, "'")}" author="tsf.compile.xosc"/>`);
  o.push("  <ParameterDeclarations>");
  o.push('    <ParameterDeclaration name="ego_speed" parameterType="double" value="12.5"/>');
  o.push('    <ParameterDeclaration name="lc_start" parameterType="double" value="1.5"/>');
  o.push('    <ParameterDeclaration name="green_s" parameterType="double" value="25"/>');
  o.push("  </ParameterDeclarations>");
  o.push("  <CatalogLocations/>");
  o.push("  <RoadNetwork>");
  o.push(`    <LogicFile filepath="${name}.xodr"/>`);
  o.push("    <TrafficSignals>");
  o.push('      <TrafficSignalController name="main_junction_plan" delay="0" reference="900">');
  o.push('        <Phase name="arm2_green" duration="17"><TrafficSignalState trafficSignalId="202" state="on"/></Phase>');
  o.push('        <Phase name="arm2_yellow" duration="3"><TrafficSignalState trafficSignalId="202" state="off;yellow"/></Phase>');
  o.push('        <Phase name="all_red" duration="2"><TrafficSignalState trafficSignalId="202" state="off"/></Phase>');
  o.push("      </TrafficSignalController>");
  o.push("    </TrafficSignals>");
  o.push("  </RoadNetwork>");
  o.push("  <Entities>");
  for (const [id, cat, color] of actors) {
    const [l, w, h] = dims[cat];
    o.push(`    <ScenarioObject name="${id}">`);
    o.push(`      <Vehicle name="${id}" vehicleCategory="${cat}">`);
    o.push(`        <BoundingBox><Center x="${f(l * 0.25, 2)}" y="0" z="${f(h / 2, 2)}"/><Dimensions width="${w}" length="${l}" height="${h}"/></BoundingBox>`);
    o.push('        <Performance maxSpeed="70" maxAcceleration="10" maxDeceleration="10"/>');
    o.push("        <Axles>");
    o.push(`          <FrontAxle maxSteering="0.5" wheelDiameter="0.7" trackWidth="${f(w * 0.85, 2)}" positionX="${f(Math.min(3, l * 0.6), 2)}" positionZ="0.35"/>`);
    o.push(`          <RearAxle maxSteering="0" wheelDiameter="0.7" trackWidth="${f(w * 0.85, 2)}" positionX="0" positionZ="0.35"/>`);
    o.push("        </Axles>");
    o.push(`        <Properties><Property name="color" value="${color}"/><Property name="tsf_category" value="${cat}"/></Properties>`);
    o.push("      </Vehicle>");
    o.push("    </ScenarioObject>");
  }
  o.push("  </Entities>");
  o.push("  <Storyboard>");
  o.push("    <Init><Actions>");
  for (const [id, , , road, v] of actors) {
    const lane = -lanes[id];
    o.push(`      <Private entityRef="${id}">`);
    o.push(`        <PrivateAction><TeleportAction><Position><LanePosition roadId="${road}" laneId="${lane}" s="${startS[id]}" offset="0"/></Position></TeleportAction></PrivateAction>`);
    o.push(`        <PrivateAction><LongitudinalAction><SpeedAction><SpeedActionDynamics dynamicsShape="step" dynamicsDimension="time" value="0"/><SpeedActionTarget><AbsoluteTargetSpeed value="${v}"/></SpeedActionTarget></SpeedAction></LongitudinalAction></PrivateAction>`);
    o.push("      </Private>");
  }
  o.push("    </Actions></Init>");
  o.push('    <Story name="uc1_story">');
  o.push('      <Act name="act_main">');
  const group = (actor: string, event: string, action: string, trigger: string) => {
    o.push(`        <ManeuverGroup maximumExecutionCount="1" name="mg_${actor}_${event}">`);
    o.push(`          <Actors selectTriggeringEntities="false"><EntityRef entityRef="${actor}"/></Actors>`);
    o.push(`          <Maneuver name="m_${event}"><Event name="${event}" priority="override" maximumExecutionCount="1">`);
    o.push(`            <Action name="a_${event}">${action}</Action>`);
    o.push(`            <StartTrigger>${trigger}</StartTrigger>`);
    o.push("          </Event></Maneuver>");
    o.push("        </ManeuverGroup>");
  };
  const simTime = (t: number) =>
    `<ConditionGroup><Condition name="t_ge_${t}" delay="0" conditionEdge="rising"><ByValueCondition><SimulationTimeCondition value="${t}" rule="greaterThan"/></ByValueCondition></Condition></ConditionGroup>`;
  group("ego", "ego_lc", '<PrivateAction><LateralAction><LaneChangeAction><LaneChangeActionDynamics dynamicsShape="sinusoidal" dynamicsDimension="time" value="4.0"/><LaneChangeTarget><RelativeTargetLane entityRef="ego" value="1"/></LaneChangeTarget></LaneChangeAction></LateralAction></PrivateAction>', simTime(1.5));
  group("ego", "ego_slow", '<PrivateAction><LongitudinalAction><SpeedAction><SpeedActionDynamics dynamicsShape="linear" dynamicsDimension="rate" value="1.5"/><SpeedActionTarget><AbsoluteTargetSpeed value="7.0"/></SpeedActionTarget></SpeedAction></LongitudinalAction></PrivateAction>', '<ConditionGroup><Condition name="stop_line_45m" delay="0" conditionEdge="rising"><ByEntityCondition><TriggeringEntities triggeringEntitiesRule="any"><EntityRef entityRef="ego"/></TriggeringEntities><EntityCondition><DistanceCondition value="45" freespace="false" rule="lessThan"><Position><LanePosition roadId="2" laneId="-1" s="190.2" offset="0"/></Position></DistanceCondition></EntityCondition></ByEntityCondition></Condition></ConditionGroup>');
  group("follow_left", "follow_gap", '<PrivateAction><LongitudinalAction><SpeedProfileAction followingMode="follow"><DynamicConstraints maxAcceleration="2.0" maxDeceleration="3.0"/><SpeedProfileEntry speed="11.0" time="0"/></SpeedProfileAction></LongitudinalAction></PrivateAction>', '<ConditionGroup><Condition name="after_ego_lc" delay="0.5" conditionEdge="rising"><ByValueCondition><StoryboardElementStateCondition storyboardElementType="event" storyboardElementRef="ego_lc" state="startTransition"/></ByValueCondition></Condition></ConditionGroup>');
  group("cross_north", "north_stop", '<PrivateAction><LongitudinalAction><SpeedAction><SpeedActionDynamics dynamicsShape="linear" dynamicsDimension="rate" value="2.0"/><SpeedActionTarget><AbsoluteTargetSpeed value="0"/></SpeedActionTarget></SpeedAction></LongitudinalAction></PrivateAction>', simTime(0));
  group("cross_south", "south_stop", '<PrivateAction><LongitudinalAction><SpeedAction><SpeedActionDynamics dynamicsShape="linear" dynamicsDimension="rate" value="1.8"/><SpeedActionTarget><AbsoluteTargetSpeed value="0"/></SpeedActionTarget></SpeedAction></LongitudinalAction></PrivateAction>', simTime(0));
  o.push('        <StartTrigger><ConditionGroup><Condition name="act_start" delay="0" conditionEdge="rising"><ByValueCondition><SimulationTimeCondition value="0" rule="greaterOrEqual"/></ByValueCondition></Condition></ConditionGroup></StartTrigger>');
  o.push("      </Act>");
  o.push("    </Story>");
  o.push('    <StopTrigger><ConditionGroup><Condition name="end" delay="0" conditionEdge="rising"><ByValueCondition><SimulationTimeCondition value="22.0" rule="greaterThan"/></ByValueCondition></Condition></ConditionGroup></StopTrigger>');
  o.push("  </Storyboard>");
  o.push("</OpenSCENARIO>");
  return o.join("\n") + "\n";
}

export function makeOsc2(name: string): string {
  return `# Generated by tsf.compile.osc2 from TSIR "${name}"
# OpenSCENARIO DSL 2.x: one serial block per actor inside a top-level parallel.
import osc.standard.*
import tsf.stdlib.junction

scenario ${name}:
    path: Path
    ego: car with:
        keep(it.color == white)
    lead_right: van with:
        keep(it.color == blue)
    follow_left: car with:
        keep(it.color == black)
    oncoming_1: car with:
        keep(it.color == red)
    oncoming_2: truck with:
        keep(it.color == white)
    cross_north: car with:
        keep(it.color == grey)
    cross_south: bus with:
        keep(it.color == yellow)

    junction: signalized_junction with:
        keep(it.arms == 4)
        keep(it.signal_plan.protected_left == true)
        keep(it.signal_plan.green == 25s)

    event ego_lc_started
    event ego_lc_done

    do parallel(duration: 22s):
        ego_track: serial:
            ego.drive() with:
                lane(lane: 2, side_of: junction.arm(2).inbound)
                speed(12.5mps)
            wait elapsed(1.5s)
            emit ego_lc_started
            ego.change_lane(num_of_lanes: 1, side: side_left_right!left, shape: sinusoidal, duration: 4s)
            emit ego_lc_done
            wait rise(ego.get_distance_to_stop_line() < 45m)
            ego.change_speed(target: 7mps, rate_profile: linear, rate_peak: 1.5mpsps)
            ego.turn(direction: left, through: junction)

        lead_track: serial:
            lead_right.drive() with:
                lane(lane: 2, side_of: junction.arm(2).inbound)
                speed(12mps)
            lead_right.change_speed(target: 12mps, rate_profile: linear, rate_peak: 1mpsps)

        follow_track: serial:
            follow_left.drive() with:
                lane(lane: 1, side_of: junction.arm(2).inbound)
                speed(11mps)
            wait @ego_lc_started
            wait elapsed(0.5s)
            follow_left.drive() with:
                keep_time_gap(target: ego, gap: 2s)

        oncoming_track: serial:
            oncoming_1.drive() with:
                lane(lane: 2, side_of: junction.arm(0).inbound)
                speed(13mps)
            oncoming_2.drive() with:
                lane(lane: 2, side_of: junction.arm(0).inbound)
                speed(11.5mps)

        cross_track: serial:
            cross_north.drive() with:
                lane(lane: 1, side_of: junction.arm(1).inbound)
                speed(8mps)
            cross_north.stop_at_stop_line(deceleration: 2mpsps)
            cross_south.drive() with:
                lane(lane: 2, side_of: junction.arm(3).inbound)
                speed(8mps)
            cross_south.stop_at_stop_line(deceleration: 1.8mpsps)

    # Oracle assertions (verified at L3)
    cover(ego.lane_changes == 1, name: ego_lc)
    cover(ego.turn == left, name: ego_left_turn)
    record(min_ttc(ego), unit: s, name: min_ttc)
    keep(no_collision())
    keep(min_ttc(ego) > 1.5s)
    keep(red_light_respected(ego))
    keep(max_decel(ego) < 4.0mpsps)
`;
}

export function makePackage(id: string, query: string, xodr: string, xosc: string, osc2: string): Record<string, unknown> {
  const trunc = (s: string, n: number) => s.split("\n").slice(0, n).join("\n") + "\n<!-- ... truncated in mock package; full file is a separate artifact -->";
  return {
    userInput: { userInput: query },
    groundTruthAnswer: {
      reasoning: {
        analysisOfUserQuery: {
          extractedInformation: {
            scenarioGoal: "Ego changes one lane to the left, then makes a protected-green left turn at a signalized 4-way intersection.",
            actors: [
              { id: "ego", type: "car", initialState: { road: 2, lane: 2, s: 140, speed_mps: 12.5 }, behavior: "lane change left (t=1.5 s, 4 s), slow to 7 m/s near stop line, left turn" },
              { id: "lead_right", type: "van", initialState: { road: 2, lane: 2, s: 115, speed_mps: 12 }, behavior: "keeps speed, straight" },
              { id: "follow_left", type: "suv", initialState: { road: 2, lane: 1, s: 175, speed_mps: 11 }, behavior: "follows ego with 2 s gap after the lane change" },
              { id: "oncoming_1", type: "car", initialState: { road: 0, lane: 2, s: 120, speed_mps: 13 }, behavior: "straight, stops at red" },
              { id: "oncoming_2", type: "truck", initialState: { road: 0, lane: 2, s: 165, speed_mps: 11.5 }, behavior: "straight, stops at red" },
              { id: "cross_north", type: "car", initialState: { road: 1, lane: 1, s: 60, speed_mps: 8 }, behavior: "stops at line" },
              { id: "cross_south", type: "bus", initialState: { road: 3, lane: 2, s: 70, speed_mps: 8 }, behavior: "stops at line (right turn)" },
            ],
            road: { type: "junction", arms: 4, lanes_per_direction: 2, speed_limit_kph: 50, signalized: true },
            environment: { time_of_day: "11:30", weather: "clear", road_surface: "dry" },
          },
          variables: [
            { name: "ego_speed", value: 12.5, unit: "mps" },
            { name: "lc_start", value: 1.5, unit: "s" },
          ],
          semanticMeaning: "Lane change in preparation for a protected left turn with surrounding traffic.",
        },
        planToDefineScenario: ["Build 4-arm junction with 2+2 lanes", "Place 7 actors", "Script ego lane change and left turn", "Add signal plan with protected left", "Attach oracle assertions"],
        requiredAssumptionsAndMissingValues: { assumptions: ["right-hand traffic", "lane width 3.5 m"], missingValues: ["NPC colours"] },
        validationAndJudging: { verifier: "tsf.verify.ladder", levels: ["L0", "L1", "L2", "L3", "L4", "L5"] },
      },
      openDriveRoadDescription: trunc(xodr, 12),
      assumptionsAndFilledMissedValues: { lane_width_m: 3.5, junction_corner_radius_m: 12, signal_phase_offset_s: 8 },
      openScenarioDescription1: trunc(xosc, 10),
      openScenarioDescription2: osc2.split("\n").slice(0, 14).join("\n") + "\n# ...",
      pythonCode: "# generated with scenariogeneration; see run artifacts",
      codeBeamerScenarioDescription: { id, status: "VERIFIED", tags: ["urban", "traffic_light", "lane_change", "left_turn", "day", "clear"] },
      derivedFunctionalRequirements: [
        "The ego shall complete the lane change before the stop line distance falls below 45 m.",
        "The ego shall not enter the junction on a red signal.",
        "Longitudinal deceleration shall not exceed 4 m/s^2.",
      ],
      scenarioVariations: [],
    },
    meta: { id, polarity: "POSITIVE", source: "text", lineage: null, label: "SYNTHETIC_GENERATION" },
  };
}

export function makeTsir(query: string): Record<string, unknown> {
  return {
    meta: {
      id: "uc1_left_turn_intersection",
      title: "Lane change left, then protected-green left turn at a signalized 4-way intersection",
      description: query,
      source: "text",
      polarity: "positive",
      tags: ["urban", "intersection", "traffic_light", "lane_change", "left_turn", "day", "clear"],
    },
    environment: { time_of_day: "11:30", weather: "clear", cloud_cover: 0.15, road_surface: "dry" },
    road: {
      type: "junction",
      signalized: true,
      signal_plan: { green_s: 25, yellow_s: 3, all_red_s: 2, protected_left: true, initial_phase_offset_s: 8 },
      corner_radius: 12,
      approaches: [0, 90, 180, 270].map((h, i) => ({
        heading_deg: h,
        length: i === 2 ? 200 : 180,
        speed_limit_kph: 50,
        inbound: { driving_lanes: 2 },
        outbound: { driving_lanes: 2 },
      })),
    },
    actors: [
      { id: "ego", role: "ego", category: "car", color: "white", initial: { road: 2, direction: "inbound", lane: 2, s: 140 }, initial_speed: 12.5, route: ["left"] },
      { id: "lead_right", category: "van", color: "blue", initial: { road: 2, direction: "inbound", lane: 2, s: 115 }, initial_speed: 12.0, route: ["straight"] },
      { id: "follow_left", category: "suv", color: "black", initial: { road: 2, direction: "inbound", lane: 1, s: 175 }, initial_speed: 11.0, route: ["left"] },
      { id: "oncoming_1", category: "car", color: "red", initial: { road: 0, direction: "inbound", lane: 2, s: 120 }, initial_speed: 13.0, route: ["straight"] },
      { id: "oncoming_2", category: "truck", color: "white", initial: { road: 0, direction: "inbound", lane: 2, s: 165 }, initial_speed: 11.5, route: ["straight"] },
      { id: "cross_north", category: "car", color: "grey", initial: { road: 1, direction: "inbound", lane: 1, s: 60 }, initial_speed: 8.0, route: ["straight"] },
      { id: "cross_south", category: "bus", color: "yellow", initial: { road: 3, direction: "inbound", lane: 2, s: 70 }, initial_speed: 8.0, route: ["right"] },
    ],
    actions: [
      { id: "ego_lc", actor: "ego", type: "lane_change", direction: "left", lanes: 1, duration: 4.0, start: { kind: "sim_time", t: 1.5 } },
      { id: "ego_slow", actor: "ego", type: "speed_change", target_speed: 7.0, accel: 1.5, start: { kind: "distance_to_stop_line", actor: "ego", distance: 45 } },
      { id: "lead_keep", actor: "lead_right", type: "speed_change", target_speed: 12.0, accel: 1.0 },
      { id: "follow_gap", actor: "follow_left", type: "follow", target: "ego", time_gap: 2.0, start: { kind: "after_action", action_id: "ego_lc", delay: 0.5 } },
      { id: "north_stop", actor: "cross_north", type: "stop_at_line", decel: 2.0 },
      { id: "south_stop", actor: "cross_south", type: "stop_at_line", decel: 1.8 },
    ],
    assertions: [
      { kind: "ego_lane_change", value: 1, label: "ego ends in the left (turn) lane" },
      { kind: "ego_turn", label: "left" },
      { kind: "no_collision" },
      { kind: "min_ttc_above", value: 1.5 },
      { kind: "red_light_respected" },
      { kind: "max_decel_below", value: 4.0 },
    ],
    duration: 22.0,
    dt: 0.05,
  };
}

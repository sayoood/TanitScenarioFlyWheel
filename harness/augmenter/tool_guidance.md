# Variation guide

## Patch format
- Output `{"lever", "rationale", "ops"}`. `ops` are applied IN ORDER to the parent TSIR shown in the prompt. Paths are JSON pointers (`/actors/2/initial_speed`); `-` as the last token appends to an array (`/actions/-`, `/actors/-`, `/static_objects/-`).
- `replace` and `remove` need an existing path; `add` creates an object key or inserts into an array. After an `add` or `remove` inside an array, later indices shift: prefer appends, and remove from the highest index to the lowest.
- A value is a complete JSON value that obeys the TSIR schema below (all enum names, units and ranges). Units are SI: speeds in m/s (km/h / 3.6), distances in m, times in s. `time_of_day` is always `HH:MM` (24 h), never a word.
- Appended actors need `id` (unique snake_case), `role` `npc`, `category`, `initial` (`road`, `direction`, `lane`, `s`), `initial_speed`, and usually `route`. Every action needs `id`, `actor`, `type`, its type fields and a `start` trigger.
- The array index tables in the prompt give the current index of every actor, action, assertion and static object.

## Weather and time of day (D1)
- Keep the environment consistent. Change `weather`, `time_of_day`, `precipitation`, `fog_visibility_m`, `cloud_cover` and `road_surface` together:
  - clear: precipitation 0, fog_visibility_m 100000, cloud_cover <= 0.25, road dry
  - cloudy: precipitation 0, cloud_cover 0.6-0.95, road dry
  - rain: precipitation 0.3-0.5, fog_visibility_m 4000-9000, cloud_cover >= 0.8, road wet
  - heavy_rain: precipitation 0.7-0.95, fog_visibility_m 800-2000, cloud_cover >= 0.9, road wet
  - fog: fog_visibility_m 60-250, precipitation <= 0.05, cloud_cover >= 0.6, road dry or wet
  - snow: precipitation 0.35-0.7, fog_visibility_m 600-3000, cloud_cover >= 0.85, road snow
- Time bands: day 08:00-17:30, dusk or dawn 06:00-08:00 and 17:30-19:30, night 19:30-06:00. "Night" means a concrete time such as `22:40`.
- Weather does not change the simulation of the other agents: keep actors, actions and the road untouched.

## Behaviour (D2)
- Change how the ego or other agents act: `initial_speed`, `route`, `headway`, start positions, and `actions` (speed_change, lane_change, follow, cut_in, stop_at_line, brake, cross, lane_offset), or append new NPC actors together with their actions.
- Realise intent with the right action:
  - "runs the red light" / "does not stop": TanitSim agents always obey traffic lights and stop lines, so this cannot be replayed literally. Realise the nearest behaviour that changes the ego's situation (the vehicle arrives much faster, brakes later, enters at the end of its green) and put the difference into `limitations`.
  - "brakes hard": a `brake` action (decel 5-7 m/s2) with a distance or time trigger, for a NPC. "The ego has to brake" means the ego's own reaction: create the cause (a slower or stopping leader, a cut-in, an oncoming or crossing vehicle that arrives earlier / faster, a pedestrian) and let the ego's car-following (IDM) react. Never script a `brake` or `stop_at_line` on the ego for this: it halts the manoeuvre and the ego intent assertions (`ego_turn`, `ego_lane_change`, `ego_reaches_s`) fail.
  - "cuts in": `cut_in` with `target`, a realistic `gap` (6-15 m) and `duration` (2-4 s).
  - "pedestrian crosses": an NPC with `category` `pedestrian` and a `cross` action, triggered by the ego's distance.
  - "faster / slower": change `initial_speed` and the `target_speed` of the speed_change actions consistently.
- Make the ego's situation more or less critical in a verifiable way. The parent's assertions are the oracle; the intent assertions (`ego_turn`, `ego_lane_change`, `red_light_respected`, `ego_reaches_s`) must still hold unless the request explicitly contradicts them. Thresholds (`min_ttc_above`, `max_decel_below`) may be retuned with `replace /assertions/<i>/value` only when the new behaviour makes the old threshold impossible; never remove an assertion.
- Do not create overlaps at t = 0: vehicles in the same lane need at least vehicle length + 5 m between them. Keep speeds plausible.

## Environment (D3)
- Where things live: on a junction the lanes of an arm are `/road/approaches/<i>/inbound/driving_lanes` and `.../outbound/driving_lanes`, its speed limit is `/road/approaches/<i>/speed_limit_kph` (a sibling of `inbound`, never inside it); on a linear road the speed limit is `/road/speed_limit_kph`. Use only keys that appear in the parent or in the schema.
- Road changes edit `/road/...`: number of lanes (`driving_lanes`, 1-6), `lane_width`, `left_turn_lane` / `right_turn_lane`, `speed_limit_kph`, `median_width`, `shoulder_width`, `segments` (line, arc, spiral with curvature in 1/m), junction `approaches`, `signalized`, `signal_plan`, `corner_radius`.
- When lanes are removed or the geometry changes, keep every actor on a valid lane and position: edit `/actors/<i>/initial` only for that, never to change behaviour.
- Static objects (`/static_objects/-`) and traffic signs use a `LanePosition` (`road`, `direction`, `lane`, `s`, `t_offset`; + = left of the lane centre, - = right): cones closing a lane are a row of `cone` objects 5-8 m apart on that lane (`t_offset` 0), signs stand at the right edge (`t_offset` about -2.5) before the obstacle. A speed-limit sign needs `sign_type` `speed_limit` and `sign_value` in km/h; keep the speed values of the agents plausible for the new limit.
- Obstacles in the ego's path must be avoidable in the replay: give the ego (or the lane) room to react, otherwise the verifier reports a collision.
- `environment.setting` is one of urban, suburban, rural, highway, industrial and should match the road.

## What the simulator (TanitSim) can and cannot do
- Every agent that follows the road obeys traffic lights and stop lines. A vehicle cannot run a red light; a request such as "the oncoming car runs the red light" is only representable approximately (a faster, later-braking or earlier-arriving vehicle, or a changed signal plan if the environment dimension is requested). Say so in `limitations` and realise the closest behaviour that really changes the ego's situation.
- Every agent follows its leader with IDM car following and a left turn without a protected phase yields to oncoming traffic; signals are fixed by the plan. A lower speed or a longer way delays the ego: if the ego no longer completes its manoeuvre inside `/duration` (verifier: "travelled only ... m"), raise `/duration` instead of changing the intent.
- Agents that never meet the ego's path do not matter: choose agents and values that interact with the ego (same lane ahead or behind, oncoming or crossing in its path, conflict points), otherwise the change has no observable effect and is rejected.
- Weather, road surface, time of day and cosmetic fields do not influence the replay; they change the rendered sensor data only.
- `limitations` is an empty string when the whole request is representable.

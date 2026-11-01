# TSIR authoring guide

- **Units:** SI throughout. Speeds are in m/s (km/h ÷ 3.6), times in s, distances in m. Right-hand traffic.

## Roads
- **`linear`** (highways, rural roads):
  - Geometry: `segments` of `line`, `arc` (curvature 1/R, + = left) or `spiral`.
  - `forward` lanes are numbered 1 = leftmost (fast lane, next to the centre/median) … N = rightmost.
  - Set `backward: null` for a one-way carriageway or divided motorway.
  - A position `s` is measured from the road start in the driving direction.
- **`junction`** (intersections, 3-5 arms):
  - Each approach has a `heading_deg` (0 = east, 90 = north, 180 = west, 270 = south). The arm extends from the junction centre in that direction.
  - Inbound lanes drive toward the centre. Lane 1 = leftmost and is the only lane allowed to turn left. The highest lane number = rightmost, the only lane allowed to turn right. Straight is allowed from all lanes.
  - Inbound `s` = distance before the stop line. Outbound `s` = distance after the junction. Keep `s` < arm length − 5.
  - `signalized: true` adds traffic lights. Opposite arms share a phase, the cycle starts with the group containing arm 0, and `protected_left` adds a leading left-turn phase.
  - Set `initial_phase_offset_s` so the ego meets the intended light state. Arrival time ≈ s / v.

## Actors
- Exactly one `role: "ego"`. Ids are short snake_case, unique.
- `category` is one of car, suv, van, truck, bus, motorcycle, bicycle, pedestrian.
- `route` = the ordered turn choices at each junction the actor passes (`left|straight|right`).
- To turn left from a right lane, the ego must first `lane_change` left. Its `route` still says `left`.

## Actions (each with `id`, `actor`, `start` trigger)
| Type | Fields |
|---|---|
| `speed_change` | `target_speed`, `accel` (comfort ≤ 2.5, emergency ≤ 8) |
| `lane_change` | `direction` left/right, `lanes`, `duration` (3-6 s typical) |
| `cut_in` | `target` (actor whose lane is entered), `gap`, `duration` (2-4 s) |
| `follow` | `target`, `time_gap` |
| `stop_at_line` | `decel` (stops at the stop line and waits for green) |
| `brake` | `decel` (hard brake to standstill) |
| `cross` | pedestrians: `speed`, `side` |

## Triggers
| Kind | Fields |
|---|---|
| `sim_time` | `t` |
| `distance_to_actor` | `actor`, `other`, `distance`, `rule`, `freespace` |
| `ttc` | `actor`, `other`, `ttc` |
| `distance_to_stop_line` | `actor`, `distance` |
| `after_action` | `action_id`, `on` start/end, `delay` |

- Use `after_action` for causal chains ("then", "immediately after").
- Use distance/TTC triggers for reactive behaviour.

## Assertions (machine-checked intent)
- Always include `no_collision`.
- Mirror every intent in the description:
  - `ego_turn` with label left/right/straight.
  - `ego_lane_change` with value = net lanes to the left (−1 = one to the right).
  - `min_ttc_above`, `red_light_respected`, `max_decel_below`, `ego_stops_before_line`, `ego_speed_between` (value, value2), `ego_reaches_s`.
- Only assert what the scripted scenario really produces. In open loop the ego follows its own actions.

## Duration, parameters, environment
- `duration` must cover the whole manoeuvre plus 3 s. Check that the ego reaches the event location in time.
- `parameters`: 2-5 logical parameters for augmentation. Each has a JSON pointer `path` into this TSIR (e.g. `/actors/0/initial_speed`) and a uniform range or a list of choices.
- `environment`: `time_of_day` HH:MM, `weather`, `road_surface`, `cloud_cover`, `precipitation`.

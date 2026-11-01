# Flywheel API v0.1 (FastAPI on thor3, `http://192.168.178.182:8080`)

All JSON. Times are in seconds or ISO-8601. The UI (operator PC) talks only to this API.

## Health and resources
`GET /api/health`
```json
{"thor": {"host": "thor3", "mem_total_gb": 122, "mem_used_gb": 41.2, "gpu_util": 0.63, "power_w": 71, "disk_free_gb": 590},
 "services": {"llm": {"up": true, "model": "qwen3.8-27b", "tok_s": 34.8}, "cosmos_sim": {"up": false}, "cosmos_transfer": {"up": false}, "scheduler": {"queue": 2, "running": "run_7f3a"}}}
```

## Runs
A run is one flywheel execution of a use case.
- `GET /api/runs?limit=50` returns a list of `RunSummary`.
- `POST /api/runs` with body `{"uc": "uc1|uc2|uc3|uc4|uc5", "input": {...}, "options": {...}}` returns a `RunSummary`.
  - uc1 input: `{"query": "..."}`
  - uc2 input: `{"video": "path-or-url"}`
  - uc3 input: `{"incident_id": "..."}`
  - uc4/uc5 input: `{"snippet": "dataset:clip_id"}`
  - options: `{"augment": {"n": 8, "strategies": ["weather","time_of_day","behavior"]}, "render": {"views": 7, "res": "480p", "frames": 121, "lidar": true}, "closed_loop": {"policy": "refe_sft4|const_vel|accel_decel", "steps": 200}}`
- `GET /api/runs/{id}` returns `RunDetail`.
- `GET /api/runs/{id}/artifacts/{name}` returns the raw file (xodr, xosc, osc, json, mp4, png).
- `GET /api/runs/{id}/scene` returns a `Scene` for the interactive 2D/3D viewer.

```ts
type Stage = {name: "analyse"|"propose"|"verify"|"repair"|"document"|"simulate"|"wsm"|"cosmos_transfer"|"augment"|"closed_loop",
              status: "pending"|"running"|"ok"|"failed"|"skipped", started?: string, seconds?: number, detail?: string};
type RunSummary = {id: string, uc: string, title: string, status: "queued"|"running"|"ok"|"failed",
                   created: string, seconds?: number, score?: number, kpis?: Record<string, number|boolean>};
type Finding = {level: "L0"|"L1"|"L2"|"L3"|"L4"|"L5", check: string, ok: boolean, message: string};
type Attempt = {index: number, passed: boolean, score: number, findings: Finding[], seconds: number};
type RunDetail = RunSummary & {input: any, stages: Stage[], attempts: Attempt[], metrics: Record<string, any>,
                 artifacts: {name: string, kind: "xodr"|"xosc"|"osc2"|"package"|"video"|"image"|"json", size: number}[],
                 cameras?: {name: string, artifact: string}[], package?: any, llm?: {calls: number, tok_per_s: number}};
type Scene = {bounds: [number, number, number, number],
              lanes: {id: string, junction: boolean, left: number[][], right: number[][], center: number[][]}[],
              stop_lines: number[][][], signals: {arm: number, x: number, y: number, states: string[]}[],
              actors: {id: string, role: "ego"|"npc", category: string, lwh: [number, number, number], color: string}[],
              t: number[], tracks: Record<string, number[][]>,   // actor -> [[x, y, yaw, v], ...] per t
              events: {t: number, type: string, id?: string, actor?: string}[]};
```

## Live events
`WS /api/events` streams `{"type": "stage"|"run"|"log"|"health", "run_id"?: string, ...payload}`.

## Harness (self-improvement)
- `GET /api/harness` returns `{"agents": [...], "knobs": {...}, "lessons": Lesson[], "revisions": Revision[], "generations": {gen: number, score: number, valid_rate: number}[]}`.
- `PUT /api/harness/knobs` with `{"agent": "lifter", "knobs": {...}}`.
- `POST /api/harness/lessons/{id}` with `{"status": "active|retired"}` (human gate).

## KPIs
`GET /api/kpis` returns `{"series": {"first_pass_valid_rate": [[iso, v], ...], "time_to_valid_s": [...], "tok_s": [...], "cosmos_s_per_frame": [...], "sim_rtf": [...]}, "latest": {...}}`.

## Config
- `GET /api/config` returns the pipeline config (rig, render defaults, augmentation strategies, closed-loop policy registry).
- `PUT /api/config` updates it.

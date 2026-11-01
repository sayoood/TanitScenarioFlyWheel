# TanitAD REFe SFT-4 as the closed-loop policy under test (thor3)

REFe SFT-4 B u400 (NAVSIM PDMS 88.25; ViT-L DINOv3 + LoRA, 322 M parameters, 4 NAVSIM pinhole cameras at 960x512, 64 proposals x 4 s)
runs as a persistent inference server in a torch container on thor3. The Flywheel side is
`src/tsf/closedloop/refe.py` (`RefePolicy`, `name = "refe_sft4"`) and `src/tsf/closedloop/rig_adapter.py` (Hyperion f-theta -> NAVSIM
pinhole reprojection). Compute is thor3 only (`ssh tsf-thor`).

```
sensor stage (Cosmos T1 / PhysicalAI-AV)         thor3 host (TSF venv)                      container tsf-refe (torch 2.15 / CUDA 13.4)
 7 Hyperion f-theta frames  ---------------->    RefePolicy                                  refe_server.py  (FastAPI, 127.0.0.1:8765)
 (1920x1080 or 1280x720)                          |- RigAdapter: LUT remap -> 4 x 512x960     |- TanitAD refe/model.py + ckpt_io.py
                                                  |- ego vector + route goal from the sim     |- model_sft4_B_u400.pt (bf16 autocast)
                                                  `- npz over HTTP  ---------------------->   `- 64 proposals -> navsim_v1_lane pick
 policy(sim, ego) -> {"traj": 8 x [x_fwd, y_left] @ 0.5 s, "dt": 0.5}  <-- resample 20 poses @ 5 Hz
```

## Start / stop

```bash
# on thor3 (or: ssh tsf-thor 'bash ~/tsf/dev/refe/tsf_repo/deploy/refe/run_server.sh')
deploy/refe/run_server.sh            # creates container tsf-refe (14 GB cgroup cap), starts the server, waits, prints /info
deploy/refe/run_server.sh stop       # stops the server and removes ONLY the tsf-refe container
curl -s http://127.0.0.1:8765/health ; curl -s http://127.0.0.1:8765/info
tail -f ~/tsf/logs/refe_server.log
```

`run_server.sh` never touches `tsf-llm` (vLLM) or `tsf-cosmos-sim`. Defaults: `REFE_ROOT=~/tsf/dev/refe`, repo =
`~/tsf/TanitScenarioFlyWheel` if it already holds `refe.py`, else `$REFE_ROOT/tsf_repo`; image `nvcr.io/nvidia/vllm:26.10-py3`
(already on the box: no new multi-GB pull).

One-time layout under `~/tsf/dev/refe/` (already done on thor3):

| path | what |
|---|---|
| `refe_src` -> `TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe` | TanitAD model code, **not vendored here** |
| `ckpt/rls1/base/model_sft4_B_u400.pt` | the SFT-4 B u400 FULL checkpoint (1,288,489,838 B, md5 `8e382eebb35b3e155862453617e5786e`) |
| `tsf_repo/` | copy of this repo's `src/`, `tests/`, `deploy/` (replace by the real checkout when it is pushed) |

**TanitAD code.** `main` of `sayoood/TanitAD` (Aug 4) does not contain the REFe package; it lives on the ship branches. Used:
`p6-ship/963638a-d9fix` @ `c1dc5d3` (sparse checkout of `.../2026-09-20-refe-plan/{refe,eval,code}`); `model.py` sha256 `e6011cd8...`, strict load of the
checkpoint passes.
```bash
git clone --depth 1 --filter=blob:none --no-checkout https://github.com/sayoood/TanitAD.git TanitAD && cd TanitAD
git fetch --depth 1 --filter=blob:none origin p6-ship/963638a-d9fix:refs/remotes/o/p6 && git sparse-checkout init --no-cone
P="/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
printf '%s\n' "$P/refe/" "$P/eval/*.py" "$P/code/" > .git/info/sparse-checkout && git checkout --detach refs/remotes/o/p6
ln -sfn "TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe" ../refe_src
```

**Checkpoint.** `eval/build_sft_ckpt.py --base model_final.pt --states scorer_states_u400.pt --arm B` builds it from the base
`Sayood/tanitad-refe-vitl16-navtrain10-grow-final/model_final.pt`, but `scorer_states_u400.pt` is **not published** in that repo (nor in any
`Sayood/*` repo). The finished artifact is: `Sayood/tanitad-refe-rls1` -> `base/model_sft4_B_u400.pt` (private, used here). Lineage was verified on thor3
with `deploy/refe/verify_ckpt_build.py`: the states file was cut from that checkpoint (78 tensors: `score_q_mlp`, `score_dec`, `score_head`), `build_sft_ckpt.py` was
run on the downloaded base (`model_final.pt` md5 `b54773f8...` matches `CHECKPOINTS_MD5.txt`) and reproduces **all 643 tensors bit-exactly**; the 565 tensors outside the scorer are
identical in base and SFT-4 (so SFT-4 = base + a 7-output scorer: `meta.n_score_components = 7`, `per_sample_calib = true`, goal mode). The file md5
differs only through serialisation/meta.

## Using the policy

```python
from tsf.closedloop import runner, refe
pol = refe.RefePolicy(server_url="http://127.0.0.1:8765")        # or TSF_REFE_URL
pol.frame_source = lambda sim, ego: render_hyperion_frames(sim)   # {camera_front_wide_120fov: HxWx3 uint8 RGB, ...}; only called when the model runs
res = runner.run(sc, m, pol)                                      # res.policy == "refe_sft4"
# or push frames yourself:  pol.set_frames(frames)  /  pol(sim, ego, frames)  /  pol.needs_frames(sim)
```
* Frames: any of the Hyperion cameras (`camera_front_wide_120fov`, `camera_cross_left_120fov`, `camera_cross_right_120fov`,
  `camera_rear_left_70fov`, `camera_rear_right_70fov`, `camera_rear_tele_30fov`; `camera_front_tele_30fov` is not used) at 1920x1080, 1280x720 or the
  1924x1084 of `hyperion8.json` (any resize of the calibrated frame), or the four NAVSIM views `CAM_F0/L0/R0/B0` directly. RGB uint8 (`frames_are_bgr=True` for cv2 frames).
* TanitSim calls the policy every tick (0.05 s). The model plans at 2 Hz (`replan_every_s = 0.5`); in between the stored plan is re-expressed in the moving ego frame
  (no frames needed). `needs_frames(sim)` tells the sensor stage when it must render. `reset()` between scenarios.
* No frames and no fallback -> `NoFramesError` (a camera-only policy must not drive blind). `fallback="const_vel"` exists for T0 smoke tests, is counted and gives up after 20 ticks.
* Model inputs (same construction as TanitAD `refe/planner.py`, i.e. the NAVSIM evaluation behind 88.25): 4 images; ego `vx vy ax ay yaw_rate steering speed` (TanitSim's kinematic
  bicycle: vy = 0, yaw_rate = v tan(delta)/2.9); **2 route goal points** at arc length S/2 and S ahead on `ego.path`, `S = max(v, 5 m/s) * 12 s`, clamped at the route end. There is no
  driving-command input in this (goal-mode) checkpoint. Output: the proposal with the highest `navsim_v1_lane` aggregate (NC*DAC*(5 EP + 5 TTC + 2 C)/12 * p(lane)), 20 poses @ 5 Hz,
  linearly resampled to 8 points @ 0.5 s. Points are **rear-axle referenced**, which is what TanitSim's pure-pursuit tracker (`_step_external`) assumes; `center_offset_m` re-references to the box centre.

## Rig adaptation (f-theta Hyperion -> 4 NAVSIM pinhole cameras)

Per virtual camera one LUT (cv2.remap maps + blend weights) per source camera and frame size, cached in memory (and in `cache_dir` if given); ~40 ms per 4-camera stack on the thor CPU.

| NAVSIM camera | pose in the ego frame (REFeConfig) | sources (tiers) |
|---|---|---|
| CAM_F0 | yaw -0.1 deg, t = (1.67, -0.02, 1.50) | `front_wide_120` |
| CAM_L0 | yaw +55.2 deg, t = (1.65, 0.14, 1.50) | `cross_left_120`, then `front_wide_120` where the first has no pixel |
| CAM_R0 | yaw -55.0 deg, t = (1.63, -0.17, 1.51) | `cross_right_120`, then `front_wide_120` |
| CAM_B0 | yaw 180 deg, t = (-0.50, 0.01, 1.45) | **blend of `rear_left_70` + `rear_right_70`** (they overlap ~14 deg behind the car; feathered by distance to the source image border, 64 px), then `rear_tele_30` as fill |

Assumptions (all documented in the module docstring; the first four are not measured):
1. **The NAVSIM camera is distorted.** nuPlan/NAVSIM JPEGs are not rectified (Caltech k1 k2 p1 p2 k3 = -0.356 0.173 -0.0021 0.00046 -0.052, fx = fy = 1545, cx = 960, cy = 560 at 1920x1080) and
   the model's 3D position embedding un-distorts every patch ray with that model. The virtual cameras therefore reproduce the barrel distortion, so a pixel looks along the ray the model assumes
   (`distort=False` = ideal pinhole, only for the `undistort=False` ablation). Output is built directly at 960x512 (intrinsics scaled by 0.5 / 512/1080).
2. **Shared vehicle frame**: nuPlan ego (rear axle) and the Hyperion vehicle frame (rear-axle ground point) are taken as the same origin with z = 0 on the road.
3. **Planar road at z = 0** (TanitSim has no pitch/roll/terrain). Each output ray is intersected with the ground plane (rays at/above the horizon or beyond 10 km are taken at 10 km = rotation only) before projecting into the
   source: exact for road surface, and for upright objects the parallax error shrinks from `baseline/depth` to `baseline*(1/depth - 1/ground_depth)`. The NAVSIM and Hyperion mounts differ by up to ~0.9 m (cross cameras), so
   rotation-only reprojection (`ground_plane=False`) is clearly worse on the road (see numbers).
4. Pixel-centre conventions differ by <= 0.5 px (ignored); frames are resizes, not crops, of the calibrated resolution; invalid pixels (outside every source) are filled with the ImageNet mean grey.
5. Coverage with `hyperion8.json` at 1920x1080: F0, L0, R0 100 %; **B0 93.8 %** (the top rows above ~18 deg elevation are outside the 70-fov rear cameras and stay grey; 97.5 % with the test clip's own calibration).
   The front tele is unused (F0's FOV lies inside the wide); rear tele only fills.
6. The PhysicalAI-AV clip rig (`prep_pai_clip.py`) uses the clip's own offline calibration: extrinsics are `qx qy qz qw x y z`, sensor -> vehicle with OpenCV camera axes (checked: front_wide z axis = +x).

## Tests and the sanity run

```bash
PYTHONPATH=src pytest tests/test_rig_adapter.py tests/test_refe_policy.py          # numpy + OpenCV (+ httpx); no GPU; 27 pass, incl. the real hyperion8.json rig on thor3
# real frames (PhysicalAI-AV chunk 2800, clip bcc52cd1-9507-425e-a5a6-5f8a795ed525; data and derived frames stay on thor3):
~/tsf/TanitScenarioFlyWheel/.venv/bin/python deploy/refe/prep_pai_clip.py --out ~/data/pai_av/derived/refe_sanity/bcc52cd1   # host: parquet, HEVC frames via ffmpeg
docker exec -w /work tsf-refe python3 /tsf/deploy/refe/sanity_pai.py --clip-dir /data/pai_av/derived/refe_sanity/bcc52cd1 \
   --ckpt /work/ckpt/rls1/base/model_sft4_B_u400.pt --refe-src /work/refe_src --variants --server-url http://127.0.0.1:8765 --out /data/pai_av/derived/refe_sanity/bcc52cd1/result.json
docker exec -w /work tsf-refe python3 /tsf/deploy/refe/sanity_pai.py ... --http-only --server-url http://127.0.0.1:8765 --out .../result_http.json   # the RefePolicy object only
```
Container note: its OpenCV has no HEVC decoder and the host ffmpeg is a hardware build without `select`/`scale`, hence `prep_pai_clip.py` grabs NV12 frames (BT.709 limited range) on the host.

Results on thor3, 10 timesteps (t = 2..11 s, 1 s apart), 4 s horizon, 8 points @ 0.5 s vs the logged future ego motion (rear axle, ego frame at t).
The route goal comes from the logged path (the dataset has no HD map); `tire_steering = atan(3.0 * curvature)` is an assumption.

| | ADE mean / median | FDE mean / median |
|---|---|---|
| REFe SFT-4 via the adapter (ground plane) | 2.40 / 1.63 m | 6.37 / 5.41 m |
| adapter rotation-only / undistorted virtual cams | 2.41 / 2.40 m | 6.52 / 6.41 m |
| constant velocity / constant curvature | 2.67 / 2.68 m | 5.48 / 5.49 m |
| oracle best of the 64 proposals | 0.70 m | |

The selection is the weak part, as in TanitAD's own NAVSIM read (pick 1.19 m, oracle 0.33 m on navtest): the proposal set contains near-human plans (0.70 m) while the pick
lands near constant velocity. This clip is an interaction clip (the lead vehicle changes the speed profile) and ADE vs the human is not PDMS.
The RefePolicy -> HTTP path reproduces the in-process numbers (ADE 2.404 / FDE 6.363 m, same proposal index at every step).

Latency and memory (thor3 GPU shared at 90-97 % utilisation with vLLM and other jobs during the measurement, so these are upper bounds):
forward **625 ms** median (613-640 ms; first call 1.3-1.8 s warm-up), adapter 43 ms, HTTP round trip 655 ms; bf16 autocast (the training numerics) vs fp32+TF32 1.41 s vs strict fp32 3.08 s, with the same pick and
<= 7 cm trajectory difference. GPU: weights 1.23 GB, peak allocated 1.50 GB, reserved 1.71 GB; server process 1.17 GiB RSS.

Adapter checks on the real frames (no image leaves the machine): L0/R0 reprojected from the cross camera vs from the front-wide camera differ by a median far-field shift of
0.06 / 0.02 px (f-theta models, extrinsics and pixel conventions are mutually consistent); on the road region the ground-plane reprojection raises the cross-camera agreement
(NCC) from 0.40 to 0.50 (L0) and from 0.20 to 0.39 (R0) over rotation-only. Max camera-to-time offset 16 ms.

## Limits and open items
* Camera frames are the only perception: the policy sees what the sensor stage renders. Cosmos renders are not NAVSIM JPEGs (domain gap); the adapter fixes geometry, not appearance.
* Goal-mode checkpoint: the route must come from the sim (`ego.path`); a driving-command checkpoint (P6) would need `nav_mode = "command"` plumbing.
* One model call per 0.5 s of sim time costs ~0.65 s wall; a 20 s scenario is ~30 s of inference.
* PhysicalAI-AV frames and everything derived from them stay on thor3 (`~/data/pai_av/derived/`); only aggregate numbers are reported.

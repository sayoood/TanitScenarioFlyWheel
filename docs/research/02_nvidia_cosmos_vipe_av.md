# Research 02: NVIDIA Cosmos 3 / WSM / ViPE / PhysicalAI-AV on Jetson Thor (as of 2026-11-01)

[S] = secondary source, [U] = unverified / inference.

## Cosmos 3 (OpenMDW-1.1)
- Family: Super 64B, Nano 16B, Edge 4B. Only **Edge** is officially listed for Jetson Orin/Thor (Thor T5000: 137 s per I2V clip).
- **`nvidia/Cosmos3-Nano-Transfer-Auto`** and **`nvidia/Cosmos3-Nano-Sim-Auto`**: V1.0, 2026-10-20, Mixture-of-Transformers, 15B BF16 (34.5 GB / 32.4 GB on HF).
  - Rig: 1-11 views of Hyperion 8.1 (7 normal + 4 fisheye 200°), plus a **128-beam LiDAR generated jointly**.
- **Transfer-Auto** (offline, bidirectional):
  - Inputs: text (also carries the camera config) and **one WSM control MP4 per camera**. LiDAR needs HD-map range-map controls. Optional first frames / one measured LiDAR sweep.
  - No depth/seg/trajectory inputs.
  - Output: 480p/720p, up to 300 frames, 10-30 fps. Quickstart: 7 cams, 201 frames, 832×480, 35 steps, guidance 3.
  - Speed (201 frames): 1 view 480p 1m22s on GB200 / 2m47s on H100; 11 views 480p 18m27s / 38m33s.
- **Sim-Auto** (closed loop): causal, few-step (**4 steps, no CFG**), distilled from Transfer.
  - Generates small chunks from cached history plus incoming WSM controls.
  - **No action input: the policy trajectory enters only via the WSM render.**
  - Speed (gen fps, H100): 1 view 480p 10.2; 7 views 1.4; 1 view + LiDAR 3.4.
  - Custom rig: post-train Transfer, then distill (TF → TF-dCM → SF-DMD) in cosmos-framework (8-GPU recipes, not on Thor).
- `nvidia/Cosmos3-Nano-Sim-Transfer`: depth/seg/edge/blur controls, causal, ≤480p, single view.
- Legacy: Cosmos-Transfer2.5-2B/auto/multiview (7 cams 720p, 65 GB VRAM, maintenance-only).
- Serving: `vllm/vllm-omni:cosmos3-aarch64` (arm64, pushed 2026-10-19). sm_110 kernels unverified.

## WSM v4 renderer + RDS-HQ
- `cosmos_auto` in https://github.com/NVIDIA/cosmos-framework (`.agents/skills/cookbooks/cosmos3/simulation/auto/wsm`): Python 3.11+, EGL/ModernGL, FFmpeg.
  - Output: 1280×720 MP4 at 30 fps in 297-frame (9.9 s) chunks.
  - **Pose rate must be an integer multiple of the output fps.**
- RDS-HQ format: `rds-hq/<component>/<clip_id>.tar`.
  - `pose`: `<frame>.pose.<cam>.npy`, 4×4 cam→world.
  - Intrinsics: `ftheta_intrinsic` [cx,cy,w,h,p0..p5,is_bw_poly,(c,d,e)] or `pinhole_intrinsic`.
  - `all_object_info` per frame: track id, type Car/Truck/Pedestrian/Cyclist/Others, is_moving, lwh, 4×4 object_to_world.
  - Map layers: `3d_lanelines` (color/style), `3d_road_boundaries`, `3d_crosswalks`, `3d_wait_lines`, `3d_poles`, `3d_traffic_signs`, `3d_road_markings`, `3d_traffic_lights` plus per-frame light states.
  - Frames: world z-up RH, origin at first-frame rear-axle ground point; vehicle FLU; camera OpenCV.
  - Helpers: `scripts/rds_hq_io.py` (`write_clip` plus validator), `retarget_hyperion.py`, skill `scenario-to-rds-hq`.
  - **No OpenDRIVE/OSC converter exists: TanitSim → RDS-HQ is our own contribution.**
- LiDAR controls: `cosmos-auto-lidar-rangemap`: 128 rings × 1800 cols, ≤105 m, Pandar128 model default, FP32 [3,T,128,1800], ~40 s per 10 s clip on RTX 6000 Ada. `range_map_to_points` converts back.
- Cosmos-Drive-Dreams (https://github.com/nv-tlabs/Cosmos-Drive-Dreams): Transfer2.5-era renderer (`render_from_rds_hq.py -s hdmap|lidar|world_scenario`).

## Pose / depth / tracking (UC2)
- **ViPE** https://github.com/nv-tlabs/vipe:
  - Versions: v1.2.0 (2026-06-09, 2.7× faster), `pose_only_long` SLAM (2026-09), `dav3` pipeline (Depth Anything 3).
  - Pinhole/wide/panorama cameras; outputs intrinsics, poses and near-metric depth (MoGe-2 prior).
  - Apache-2.0 (Unik3D NC). Source build of CUDA extensions (cu128 reference); no aarch64 mention.
- Depth Anything 3: DA3METRIC-LARGE / Base / Small are Apache-2.0 (Giant / -1.1 are NC).
- SAM 3 / 3.1: SAM License, 848M, PyTorch 2.7+.
- MapAnything apache-v1 (metric, static).
- VGGT-Ω (2026-05; license unclear). Pi3X is NC. MegaSaM (CUDA 11.8) is a poor fit.
- Dash2Sim (arXiv 2606.07366): SAM3 + WildDet3D + EKF. Metric scale needs anchoring.

## PhysicalAI-Autonomous-Vehicles (gated, NVIDIA AV Dataset License: 12-month term, no redistribution/derivatives)
- Scale: 306,152 clips × 20 s, 25 countries.
- **7 cameras, 1920×1080 at ~30 fps, f-theta** (`fw_poly_0..4`, `bw_poly_0..4`):

  | Camera | FOV |
  |---|---|
  | front_wide | 120° |
  | front_tele | 30° |
  | cross_left | 120° |
  | cross_right | 120° |
  | rear_left | 70° |
  | rear_right | 70° |
  | rear_tele | 30° |

- **lidar_top_360fov**: 128 ch, 10 Hz, parquet + Draco (x, y, z, t, intensity).
- Egomotion ~100 Hz in an anchor frame (origin/yaw zeroed at t=0). Camera frame is OpenCV.
- Extrinsics: quaternion + xyz. The direction label is ambiguous, so verify numerically.
- `pip install physical_ai_av` (0.2.3): `get_clip_feature(clip_id, feature, maybe_stream=True)` streams one clip.
- Alpamayo 1.5 (10B), Alpamayo2-Super (34B). The `alpamayo_aligned` split has 1,536 AlpaSim clips.

## Cosmos-based closed loop (not NuRec/3DGS)
- Sim-Auto itself.
- **OmniDreams** (arXiv 2606.03159, nv-tlabs/omni-dreams, Apache-2.0 code, gated weights): single front view 704×1280, ~52 FPS on GB300, AlpaSim gRPC. FlashDreams runtime ([S] ≥80 GB, CUDA 13).
- Community: `cosmos3-canvas` (Sim-Auto behind CARLA, live WSM v4 + range maps); `roadgen` (OpenDRIVE + RDS-HQ export).

## Thor risks (drive the design)
1. No official Thor support for the Auto models (only Edge). The arm64 vLLM-Omni image has unknown sm_110 kernels.
2. Not real time on Thor. Treat Sim-Auto closed loop as **sim-time-decoupled** (the world pauses while frames generate) and Transfer-Auto as offline batch.
3. Triton/ptxas sm_110a issues were reported. JetPack 7.2 SBSA driver lets NGC multi-arch images run.
4. No post-training on Thor.
5. Memory: ~30 GB of weights plus activations in 122 GB shared with the LLM, so GPU time-sharing is required.
6. EGL on the Jetson iGPU is untested for the WSM renderer.
7. Licensing: PhysicalAI-AV no-derivatives / 12-month term; NC checkpoints are to be avoided.

# ViPE review: functions, reported results, and gaps in our use

Sources:

- The repository [nv-tlabs/vipe](https://github.com/nv-tlabs/vipe): README, `configs/`, `vipe/` sources and release notes.
- The paper "ViPE: Video Pose Engine for 3D Geometric Perception" (arXiv 2508.10934).
- Our integration: `deploy/vipe/*`, `src/tsf/ingest/{vipe,npc}.py`, `src/tsf/agents/video.py`.

Our pinned commit `8c9f361` (2026-09-10) is the latest upstream. It includes v1.2.0, the GPU bundle-adjustment solver and `pose_only_long`.

## 1. What ViPE does

**Pipeline**

1. Intrinsics are initialised with GeoCalib on a few early frames.
2. Instance masks are built from GroundingDINO boxes, SAM ViT-B and an AOT tracker. Detection is re-run every `init.instance.kf_gap_sec`, 2 s by default. The sky is masked as well.
3. A DROID-style dense SLAM runs flow bundle adjustment at 1/8 resolution, keyframe selection and a back-end bundle adjustment. Intrinsics are refined in the back end.
4. A monocular metric-depth prior regularises the bundle adjustment and anchors metric scale. `slam.keyframe_depth` is one of `moge2-l` (the pose_only default), `unidepth-l`, `dav3` or `metric3d-small`.
5. Optionally, `pipeline.post.depth_align_model` produces dense depth. `adaptive_<metric>_<svda|vda>` aligns Video-Depth-Anything to the SLAM map, which gives temporally consistent metric depth.

**Input size.** SLAM resizes every frame to about 384x512 pixels of area. Feeding 480p therefore adds nothing to pose accuracy.

**Camera models**

- Pinhole: a single focal length and a centred principal point.
- `mei` wide-angle: one distortion term.
- Panorama: six pinhole views.

**Pipelines** (`configs/pipeline/`): default, dav3, lyra, no_vda, static_vda, wide_angle, panorama, pose_only, pose_only_long. `pose_only_long` streams frames and closes with a Sim(3) pose graph, which needs gtsam.

**Main options**

- `slam.ba.solver` (legacy, fused_v2) and `slam.ba.fused`
- `slam.ba.robust_kernel` (none, huber, tukey, gnc_tls)
- `slam.optimize_intrinsics`
- `slam.sparse_tracks.name` (dummy, cuvslam)
- `pipeline.init.intrinsics` (geocalib, gt)

**Outputs**

- `pose/*.npz`: camera-to-world 4x4 matrices in OpenCV convention.
- `intrinsics/*.npz`: [fx, fy, cx, cy].
- `depth/*.zip`: fp16 EXR z-depth.
- `mask/*.zip`: instance ids, with phrases in `mask/*.txt`.
- Optionally, the SLAM point map.

`pose_only` writes poses and intrinsics only.

## 2. Reported accuracy and speed (paper §4)

| Benchmark | ATE | Focal error | Comparison |
| --- | --- | --- | --- |
| TUM freiburg1 | 3.6 cm | 1.8° | |
| TUM freiburg3 | 1.5 cm | 0.6° | |
| KITTI (first 1024 frames) | 9.2 m | 1.9° | VGGT 23.8 m, MegaSAM 25.4 m |
| RDS, 64 driving sequences | 5.0 m | 7.9° | VGGT is better on focal error (5.9°) |

- **Speed:** 3-5 FPS at 640x480 on an RTX 5090. On the Thor we measure about 1 FPS end to end: 300 frames in 310 s, with cuDNN disabled, masks enabled and model loading included.
- **Not covered by the paper:**
  - The experiments use Metric3Dv2, not the MoGe-2 we use.
  - Metric scale is described only qualitatively ("roughly metric").
  - Night and fisheye footage are not evaluated, and there is no limitations section.
- **Off in every shipped configuration:** the paper's cuVSLAM sparse-track term (`sparse_tracks: dummy`).

## 3. Gaps in our use

| Area | Gap | Lever |
| --- | --- | --- |
| Ego-motion accuracy | Bundle adjustment has no robust kernel, so moving objects that escape the masks bias the result | `slam.ba.robust_kernel=huber` or `gnc_tls` |
| Ego-motion accuracy | The ego-only run and the dense run use different mask phrases, so their ego tracks differ | One phrase set for both |
| Ego-motion accuracy | 20 s windows are chained rigidly; each window gets its own MoGe-2 scale | Estimate a per-window scale from the 2 s overlap (Sim(3)) |
| Ego-motion accuracy | The camera model assumes no distortion and a centred principal point | Rectify cropped or distorted footage first |
| Dense depth | Our separate per-frame MoGe-2 pass flickers by a few percent | `pipeline.post.depth_align_model=adaptive_moge2-l_svda` (SLAM-anchored, temporally consistent) |
| Other road users | Only 2D masks; new vehicles are detected only every 2 s; ids reset per window | `kf_gap_sec` 0.5-1.0; `pose_only_long` for stable ids |
| Speed on Thor | Untried settings; models are reloaded per window | `slam.ba.solver=fused_v2`, `slam.ba.fused=true`, 360p input, resident models |
| Licences | UniDepth, Video-Depth-Anything-Large and DA3-GIANT weights are non-commercial; the TrackAnything code is AGPL-3.0 | Keep internal; check before any product use |

## 4. Pseudo-LiDAR from ViPE depth

**Method**

1. Unproject the z-depth with the intrinsics: X_c = d·[(u−cx)/fx, (v−cy)/fy, 1].
2. Transform to the world frame with the camera-to-world pose.
3. Level the cloud with the up axis before drawing a bird's-eye view.

**Cleaning**

- Drop sky pixels and depth-0 pixels.
- Drop instance pixels from the static accumulation, since moving objects get only single-view depth.
- Cut at about 80 m.
- Filter flying pixels at object edges.

**Scale**

- Poses and depth share MoGe-2's metric head, so they are biased together.
- A 5 % focal error becomes a 5 % range error, so pass the focal length when it is known.
- The 1.2-1.6 m mount-height prior leaves about ±15 % scale uncertainty.
- Apply one scale factor to poses, depth and tracks together.

## 5. Recommendations, by gain per effort

1. **Sweep configurations on the PhysicalAI-AV ground-truth harness:** `fused_v2`, a robust kernel, and 360p SLAM input. These take hours, and may improve both speed and robustness.
2. **Align scale across windows** using the overlap. This is about 20 lines.
3. **Switch dense depth to ViPE's own `adaptive_moge2-l_svda` post step**, after a licence check.
4. **Improve recall of other road users:** set `kf_gap_sec` to 0.5-1.0 and use one phrase set.
5. **Use `pose_only_long` for clips over 20 s.** It needs gtsam on aarch64; a small scipy pose graph would replace it if no wheel exists.

Recommendations 1, 3 and 4 were forwarded to the ViPE workstream on 2026-11-01.

# Cosmos3-Nano Auto on Thor (rendering stage)

WSM control videos (from `tsf.cosmos.wsm`) -> camera video with NVIDIA **Cosmos3-Nano-Transfer-Auto** (offline, 35 steps) or
**Cosmos3-Nano-Sim-Auto** (causal, 4 steps, 0.4 s chunks). Served by vLLM-Omni in `vllm/vllm-omni:cosmos3-aarch64` (torch 2.13 / CUDA 13,
FlashAttention-4 CuTe kernels JIT-compile fine on sm_110). Everything runs on `tsf-thor` (thor3) only.

Work dir `~/tsf/dev/cosmos_gen/` (mounted at `/work`): `deploy/sim_auto_thor.yaml`, `patches/pipeline_cosmos3_nano_sim_auto.py`,
`cache/` (FA4 / triton JIT caches), `out/<scenario>/<run>/`. Logs `~/tsf/logs/cosmos_gen_*.log`. Weights: `~/.cache/huggingface` (offline).

## Rules on thor3 (122 GB unified memory, shared with `tsf-llm` ~43 GB)

* One Cosmos server at a time (`tsf-cosmos-sim` or `tsf-cosmos-transfer`, port 8100). Stop it when done: `docker rm -f tsf-cosmos-sim`.
* `free -g` first: Sim-Auto needs ~35 GB idle and +5..30 GB while generating; Transfer-Auto ~40 GB idle.
* `deploy/watchdog.sh 6 <container> &` removes *our* container if `MemAvailable < 6 GB` (start it in the background before a run).
* Never touch `tsf-llm`, port 8000 or 8080.

## Sim-Auto patch (fits next to the LLM)

Stock vLLM-Omni pre-sizes the paged KV ring for the worst case, 7 views + LiDAR = **27 GB**, and refuses to start otherwise.
`sim_auto_kv_views.patch` adds `TSF_KV_VIEWS=N` (ring for N RGB-only views: 1 view ~4 GB, 3 views ~8 GB). RGB-only; no LiDAR.

```bash
# on thor, once
cd ~/tsf/dev/cosmos_gen && mkdir -p patches cache/fa_cute cache/triton_sim cache/triton_transfer
cid=$(docker create --entrypoint "" vllm/vllm-omni:cosmos3-aarch64 /bin/true)
docker cp $cid:/usr/local/lib/python3.12/dist-packages/vllm_omni/diffusion/models/cosmos3_nano_sim_auto/pipeline_cosmos3_nano_sim_auto.py patches/
docker rm $cid
# from the repo: scp deploy/cosmos/{sim_auto_thor.yaml,sim_auto_kv_views.patch,watchdog.sh} tsf-thor:~/tsf/dev/cosmos_gen/deploy/
patch patches/pipeline_cosmos3_nano_sim_auto.py < deploy/sim_auto_kv_views.patch
```

## Serve

```bash
# Sim-Auto (views = max cameras per request)
python - <<'EOF'
from tsf.cosmos import generate as g
g.serve("sim", kv_views=3); print(g.wait_ready("sim"), "s")      # ~5 min: 27 GB weights from NVMe
EOF
# Transfer-Auto (regional torch.compile flags are required by the checkpoint)
python -c "from tsf.cosmos import generate as g; g.serve('transfer'); print(g.wait_ready('transfer'))"
```

Equivalent docker command (Sim-Auto; Transfer-Auto replaces the model/class and the last flags with
`nvidia/Cosmos3-Nano-Transfer-Auto`, `Cosmos3MultiviewPipeline`, `--diffusion-compile-granularity regional --no-diffusion-compile-dynamic`):

```bash
docker run -d --name tsf-cosmos-sim --runtime nvidia --gpus all --ipc=host \
  -e HF_HUB_OFFLINE=1 -e VLLM_OMNI_VIDEO_SYNC_TIMEOUT=3600 -e VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=3600 -e TSF_KV_VIEWS=3 \
  -e FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 -e FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/work/cache/fa_cute \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface -v $HOME/tsf/dev/cosmos_gen:/work -v $HOME/tsf/dev/cosmos_gen/cache/triton_sim:/root/.triton \
  -v $HOME/tsf/dev/cosmos_gen/patches/pipeline_cosmos3_nano_sim_auto.py:/usr/local/lib/python3.12/dist-packages/vllm_omni/diffusion/models/cosmos3_nano_sim_auto/pipeline_cosmos3_nano_sim_auto.py:ro \
  -p 127.0.0.1:8100:8000 -w /app/vllm-omni vllm/vllm-omni:cosmos3-aarch64 \
  vllm serve nvidia/Cosmos3-Nano-Sim-Auto --revision main --omni --model-class-name Cosmos3NanoSimAutoPipeline \
  --deploy-config /work/deploy/sim_auto_thor.yaml --tensor-parallel-size 1 --no-guardrails --host 0.0.0.0 --port 8000 --init-timeout 1800
```

`sim_auto_thor.yaml` = the image's `cosmos3_nano_sim_auto.yaml` with `gpu_memory_fraction: 0.2`.

## Generate

Run with the repo venv on thor (`httpx`, `numpy`, `cv2`; ffmpeg with libx264 is taken from `imageio-ffmpeg`, Thor's apt ffmpeg only has NVENC).

```python
from pathlib import Path
from tsf.cosmos import generate as g
wsm = Path("~/tsf/TanitScenarioFlyWheel/runs/wsm/uc1_left_turn_intersection").expanduser()
cams = ["camera_front_wide_120fov", "camera_cross_left_120fov", "camera_cross_right_120fov"]

# Sim-Auto, WSM-only text2video (no RGB needed); captions=None -> deterministic template from the scenario id
r = g.sim_chunk(wsm, "out/uc1/sim", cams, frames=97, res="480", captions=None)
# Closed-loop style chunks: window k+1 starts from the last generated frame of window k (I2V)
r = g.sim_rollout(wsm, "out/uc1/roll", cams[:1], chunks=6, chunk_frames=49)
# Transfer-Auto (offline), after g.stop("sim"); g.serve("transfer")
r = g.transfer(wsm, "out/uc1/transfer", cams[:1], frames=29, res="480", captions="captions/")   # dir of <camera>.json sidecars
g.compare_mosaic(r["controls"], r["videos"], "out/uc1/mosaic.mp4", title="UC1")
```

Result dict: `videos`, `request_s`, `s_per_frame_view`, `mem_used_peak_gb` (system used, unified memory), `last_frames` (Sim only).
`frames` must be 4k+1; `res` is `"480"` (832x480) or `"720"` (1280x720). Captions: `tsf.cosmos.captions.template_captions` (placeholder
bodies, 300-400 words, timestamped) or LLM-written `captions/<camera_key>.json`
(`{"caption_structured": {"chunk_0_297": {"start_frame": 0, "end_frame": 297, "caption": "..."}}}`).

## Known limits

* Sim-Auto needs the camera set in the request <= `kv_views`; changing `kv_views` means a server restart (~5 min).
* vLLM-Omni Sim-Auto has no tick API ("continuing a live session ... not supported yet"): one request = one rollout; `sim_rollout` chains
  requests through RGB first-frame conditioning (no KV carry-over between chunks).
* Joint LiDAR needs the 27 GB ring (7 views + LiDAR) and does not fit next to `tsf-llm`.

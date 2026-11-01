#!/usr/bin/env bash
# Start (idempotently) the REFe SFT-4 inference server on thor3 (tsf-thor). See README.md.
#   deploy/refe/run_server.sh            start container + server, wait until healthy, print /info
#   deploy/refe/run_server.sh stop       stop the SERVER process and remove ITS container (never touches tsf-llm / tsf-cosmos-sim)
#   NAME=tsf-refe PORT=8765 REFE_ROOT=~/tsf/dev/refe TSF_REPO=~/tsf/TanitScenarioFlyWheel deploy/refe/run_server.sh
#
# Layout it expects under $REFE_ROOT (default ~/tsf/dev/refe):
#   refe_src/                       TanitAD ".../2026-09-20-refe-plan/refe" (model.py, ckpt_io.py)  -> README "TanitAD code"
#   ckpt/rls1/base/model_sft4_B_u400.pt   the SFT-4 B u400 FULL checkpoint                           -> README "Checkpoint"
set -euo pipefail

NAME="${NAME:-tsf-refe}"
PORT="${PORT:-8765}"
REFE_ROOT="${REFE_ROOT:-$HOME/tsf/dev/refe}"
if [[ -z "${TSF_REPO:-}" ]]; then
  if [[ -f "$HOME/tsf/TanitScenarioFlyWheel/src/tsf/closedloop/refe.py" ]]; then TSF_REPO="$HOME/tsf/TanitScenarioFlyWheel"; else TSF_REPO="$REFE_ROOT/tsf_repo"; fi
fi
CKPT="${CKPT:-/work/ckpt/rls1/base/model_sft4_B_u400.pt}"
IMAGE="${IMAGE:-nvcr.io/nvidia/vllm:26.10-py3}"          # already on the box: CUDA-13 / aarch64 PyTorch for Thor
LOGDIR="${LOGDIR:-$HOME/tsf/logs}"
MEM="${MEM:-14g}"                                         # cgroup cap: the host's 122 GB are shared with vLLM + Cosmos
mkdir -p "$LOGDIR" "$REFE_ROOT/home" "$REFE_ROOT/pylibs"

if [[ "${1:-}" == "stop" ]]; then
  docker exec "$NAME" sh -c 'kill $(cat /work/home/refe_server.pid)' 2>/dev/null || true
  docker rm -f "$NAME" >/dev/null 2>&1 && echo "removed $NAME" || echo "$NAME not present"
  exit 0
fi

[[ -f "$REFE_ROOT/refe_src/model.py" ]] || { echo "missing $REFE_ROOT/refe_src/model.py (TanitAD refe/ directory)"; exit 2; }
[[ -f "$REFE_ROOT${CKPT#/work}" ]] || { echo "missing checkpoint $REFE_ROOT${CKPT#/work}"; exit 2; }

if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  free -g | awk 'NR==2{print "host memory: " $7 " GB available"}'
  docker run -d --name "$NAME" --runtime nvidia --ipc host --network host --user "$(id -u):$(id -g)" \
    -e HOME=/work/home -e HF_HOME=/hf -e PYTHONPATH=/tsf/src:/work/pylibs \
    --memory "$MEM" --memory-swap "$MEM" --entrypoint sleep \
    -v "$REFE_ROOT":/work -v "$TSF_REPO":/tsf:ro -v "$HOME/.cache/huggingface":/hf -v "$HOME/data":/data -v "$LOGDIR":/logs \
    "$IMAGE" infinity >/dev/null
fi

if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
  docker exec -d -w /work "$NAME" sh -c "echo \$\$ > /work/home/refe_server.pid; exec python3 /tsf/deploy/refe/refe_server.py \
      --ckpt $CKPT --refe-src /work/refe_src --host 127.0.0.1 --port $PORT > /logs/refe_server.log 2>&1"
  for _ in $(seq 1 90); do sleep 2; curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; done
fi
curl -sf "http://127.0.0.1:$PORT/info" && echo || { echo "server did not come up, see $LOGDIR/refe_server.log"; exit 1; }

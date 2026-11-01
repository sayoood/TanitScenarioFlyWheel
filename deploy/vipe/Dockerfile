# ViPE (nv-tlabs/vipe) for Jetson AGX Thor (sm_110, JetPack 7.2, CUDA 13.x, aarch64).
# Derived from the vLLM NGC image already on thor3: it ships the Thor-built PyTorch and nvcc, so no extra
# multi-GB pull. Build context = ViPE source checkout (see setup.sh).
FROM nvcr.io/nvidia/vllm:26.10-py3 AS deps

ENV DEBIAN_FRONTEND=noninteractive

RUN uv venv --seed --system-site-packages /opt/venv
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:${PATH}

# torch/torchvision/numpy come from the base image (satisfy the requirements, never reinstalled).
# Skipped: gtsam (only the pose_only_long pipeline imports it; no matching aarch64 wheel), ray (never imported).
# The big wheels (rerun-sdk 166 MB, imageio-ffmpeg, opencv, transformers) are prefetched by
# fetch_weights.sh wheels and passed as the named build context 'wheels': pip's single stream is too slow on
# the shared link. The pip cache mount makes retries cheap. PIP_NO_BUILD_ISOLATION=1 in the base image does
# NOT disable isolation (pip quirk), so --no-build-isolation is explicit below: ViPE must build against the
# image's Thor torch, not a PyPI one.
RUN --mount=type=cache,target=/root/.cache/pip --mount=type=bind,from=wheels,target=/wheels \
    pip install /wheels/*.whl click einops gdown huggingface_hub hydra-core kornia matplotlib ninja \
    omegaconf "OpenEXR<3.3.0" "pydantic>=2,<3" pillow python-pycg \
    safetensors scipy timm tqdm "transformers>=4,<5" viser

FROM deps
# sm_110 only (no GPU during build); MAX_JOBS caps nvcc/cicc (~6 GB each) on the shared 122 GB box.
ARG TORCH_CUDA_ARCH_LIST=11.0
ARG MAX_JOBS=2
COPY . /opt/vipe-src
RUN --mount=type=cache,target=/root/.cache/pip cd /opt/vipe-src \
    && TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} MAX_JOBS=${MAX_JOBS} \
       pip install --no-build-isolation --no-deps -v . \
    && rm -rf /opt/vipe-src/build

ENV TORCH_HOME=/cache/torch HF_HOME=/cache/hf
COPY --from=tsf vipe_worker.py /opt/tsf/vipe_worker.py
WORKDIR /work

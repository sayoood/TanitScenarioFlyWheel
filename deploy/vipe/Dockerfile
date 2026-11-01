# ViPE (nv-tlabs/vipe) for Jetson AGX Thor (sm_110, JetPack 7.2, CUDA 13.x, aarch64).
# Derived from the vLLM NGC image already on thor3: it ships the Thor-built PyTorch and nvcc, so no extra
# multi-GB pull. Build context = ViPE source checkout (see setup.sh).
FROM nvcr.io/nvidia/vllm:26.10-py3

ARG TORCH_CUDA_ARCH_LIST=11.0
ARG MAX_JOBS=6
ENV DEBIAN_FRONTEND=noninteractive

RUN uv venv --seed --system-site-packages /opt/venv
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:${PATH}

# torch/torchvision/numpy come from the base image (satisfy the requirements, never reinstalled).
# gtsam is only imported by the pose_only_long pipeline and has no matching aarch64 wheel: skipped.
RUN pip install click einops gdown huggingface_hub hydra-core "imageio[ffmpeg]" kornia matplotlib ninja \
    omegaconf "opencv-python-headless<5" "OpenEXR<3.3.0" "pydantic>=2,<3" pillow python-pycg ray \
    rerun-sdk safetensors scipy timm tqdm "transformers>=4,<5" viser

COPY . /opt/vipe-src
RUN cd /opt/vipe-src \
    && TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} MAX_JOBS=${MAX_JOBS} pip install --no-deps -v . \
    && rm -rf /opt/vipe-src/build

ENV TORCH_HOME=/cache/torch HF_HOME=/cache/hf
COPY vipe_worker.py /opt/tsf/vipe_worker.py
WORKDIR /work

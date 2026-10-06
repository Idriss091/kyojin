# Kyojin server image for AMD Strix Halo (gfx1151). Works with docker and podman.
#   docker build -f docker/Dockerfile -t kyojin .
# Follows the README Quickstart: ROCm torch + rocm-sdk-devel from AMD's gfx1151 index, requirements.txt,
# `rocm-sdk init`, tools/strix_halo/env.sh, build.sh. The models are NOT in the image: mount them.

FROM ubuntu:24.04
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
# Single stage on purpose: a second stage would copy the ~14 GB tree again and double the disk peak.
# g++, libc6-dev, python3-dev stay: the servers compile a few HIP kernels (hipcc) and Triton kernels at start.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev gcc g++ libc6-dev make git ca-certificates \
        libnuma1 libatomic1 libgomp1 libdrm2 libelf1 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/kyojin
RUN python3 -m venv .venv
ENV PATH=/opt/kyojin/.venv/bin:$PATH
COPY requirements.txt .
COPY . .
# One layer on purpose: the pip downloads and the unpruned SDK tree never reach an image layer.
# The SDK (hipcc, headers) stays, pruned of test binaries, profiler data and client samples.
RUN pip install --pre torch rocm-sdk-devel --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
    && pip install -r requirements.txt \
    && rocm-sdk init \
    && export EXL3_ROCM_SDK=$(rocm-sdk path --root) \
    && source tools/strix_halo/env.sh \
    && ./build.sh \
    && python -c "import torch, exllamav3_ext" \
    && rm -rf build .rebuild.* .sdk-deps /root/.cache \
    && cd "$EXL3_ROCM_SDK" && rm -rf clients share/rocprofiler-sdk share/hipblaslt share/rocdecode share/rocsparse bin/*gtest* bin/*-test bin/*-bench bin/*-gemm-tune bin/*test*
ENV EXL3_ROCM_SDK=/opt/kyojin/.venv/lib/python3.12/site-packages/_rocm_sdk_devel
ENV MODEL_DIR=/models PORT=8000 HOME=/root
EXPOSE 8000
VOLUME ["/models"]
HEALTHCHECK --start-period=900s --interval=30s --timeout=5s --retries=3 \
    CMD curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null || exit 1
STOPSIGNAL SIGTERM
ENTRYPOINT ["/opt/kyojin/docker/entrypoint.sh"]

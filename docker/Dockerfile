# Kyojin server image for AMD Strix Halo (gfx1151). Works with docker and podman.
#   docker build -f docker/Dockerfile -t kyojin .
# Follows the README Quickstart: ROCm torch + rocm-sdk-devel from AMD's gfx1151 index, requirements.txt,
# `rocm-sdk init`, tools/strix_halo/env.sh, build.sh. The models are NOT in the image: mount them.

FROM ubuntu:24.04 AS build
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev gcc g++ make git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/kyojin
RUN python3 -m venv .venv
ENV PATH=/opt/kyojin/.venv/bin:$PATH
COPY requirements.txt .
COPY . .
# One layer on purpose: the 12 GB SDK tree (needed only to compile) never reaches an image layer.
RUN pip install --pre torch rocm-sdk-devel --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
    && pip install -r requirements.txt \
    && rocm-sdk init \
    && export EXL3_ROCM_SDK=$(rocm-sdk path --root) \
    && source tools/strix_halo/env.sh \
    && ./build.sh \
    && python -c "import torch, exllamav3_ext" \
    && rm -rf "$EXL3_ROCM_SDK" build .rebuild.* .sdk-deps /root/.cache

FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 libnuma1 libatomic1 libgomp1 libdrm2 libelf1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/kyojin /opt/kyojin
WORKDIR /opt/kyojin
ENV PATH=/opt/kyojin/.venv/bin:$PATH \
    EXL3_HSA_LIB=/opt/kyojin/.venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1
ENV MODEL_DIR=/models PORT=8000 HOME=/root
EXPOSE 8000
VOLUME ["/models"]
HEALTHCHECK --start-period=900s --interval=30s --timeout=5s --retries=3 \
    CMD curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null || exit 1
STOPSIGNAL SIGTERM
ENTRYPOINT ["/opt/kyojin/docker/entrypoint.sh"]

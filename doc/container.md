# Running Kyojin in a container (docker / podman)

One image serves one model. Every setting is an environment variable. The model folder is mounted, never baked in. The GPU is passed in as `/dev/kfd` and `/dev/dri`. Host requirements: a gfx1151 machine (Strix Halo, 128 GB) with the `amdgpu` kernel driver. No ROCm install is needed on the host.

## Build

```bash
docker build -f docker/Dockerfile -t kyojin .       # or: podman build -f docker/Dockerfile -t kyojin .
```

The build follows the README Quickstart: ROCm torch and `rocm-sdk-devel` from AMD's gfx1151 index, `requirements.txt`, `rocm-sdk init`, `tools/strix_halo/env.sh`, `build.sh`. It needs no GPU. The SDK tree is dropped in the final stage. The torch wheel comes from AMD's nightly index, so two builds on different days can carry different torch versions.

## Run

```bash
docker run --rm --name kyojin \
  --device /dev/kfd --device /dev/dri --group-add "$(getent group video | cut -d: -f3)" --group-add "$(getent group render | cut -d: -f3)" \
  --security-opt seccomp=unconfined --ipc=host \
  -v /path/to/Qwen3.8-Flash-Next-EXL3-Yamz:/models:ro \
  -e KYOJIN_MODEL=qwen -e CONTEXT=32768 -e PORT=8000 -p 8000:8000 \
  kyojin
```

Not tested (no podman on the build machine):

```bash
podman run --rm --name kyojin \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt label=disable --ipc=host \
  -v /path/to/Qwen3.8-Flash-Next-EXL3-Yamz:/models:ro \
  -e KYOJIN_MODEL=qwen -e CONTEXT=32768 -e PORT=8000 -p 8000:8000 \
  kyojin
```

Then `curl http://localhost:8000/health` and `python tools/bench.py --base http://localhost:8000/v1` from the host. The port stays closed while the model loads. First start of a new container: about 6 minutes for Qwen (measured 370 s), because the dense GEMM kernels are compiled and cached under `/root/.cache` and a fresh container has an empty cache. The same start on bare metal with a warm cache takes about 1 minute. Keep the cache between runs with a named volume, `-v kyojin-cache:/root/.cache` (not yet measured). GLM tunes its kernels for about 10 minutes on a first launch (see README); keep the container, or mount a volume on `/root/.cache`, between runs if you want that kept.

The image has no `video` or `render` group by name, so docker needs the numeric group ids of the host (`getent group video render`). In the llama-swap example below, replace `${VIDEO_GID}` and `${RENDER_GID}` by those numbers.

## Variables

| Variable | Default | Meaning |
|---|---|---|
| `KYOJIN_MODEL` | `auto` | `qwen`, `glm`, `mimo`, or `auto` (read from `config.json` of the pack) |
| `MODEL_DIR` | `/models` | The pack folder, or a folder holding exactly one pack |
| `HOST` | `0.0.0.0` | Listen address inside the container |
| `PORT` | `8000` | Listen port (also used by the health check) |
| `CONTEXT` | server default | KV cache size in tokens (`-c`) |
| `KYOJIN_MODEL_ID` | server default | Id shown by `/v1/models` and expected in requests |
| `KYOJIN_DEFAULT_TEMPERATURE`, `_TOP_P`, `_TOP_K`, `_MIN_P` | server default | Sampling used when a request omits it (`TOP_K` and `MIN_P`: Qwen only) |
| `KYOJIN_REASONING_EFFORT` | server default | `low`, `medium`, `xhigh` (Qwen, GLM) |
| `KYOJIN_DRAFT_TOKENS` | server default | Draft tokens per round (`--ndt`, or `--num-draft` on GLM) |
| `KYOJIN_DRAFT_POLICY` | server default | `mix`, `mtp`, `off` (Qwen) |
| `KYOJIN_NO_VISION`, `KYOJIN_NO_THINKING` | off | Set `1` to enable (Qwen) |
| `KYOJIN_NO_UNCENSOR` | off | Set `1` to enable (Qwen, GLM) |
| `KYOJIN_SLOT_SAVE_PATH` | `~/cache/llama-slots` | Prompt-slot save folder; mount a volume to keep it |
| `KYOJIN_EXTRA_ARGS` | empty | Extra `serve.py` arguments, shell-quoted: `--default-max-tokens 512` |
| `EXL3_*` | unset | Any engine variable is passed through unchanged |

Arguments after the image name are appended too: `docker run ... kyojin --default-max-tokens 512`. An option the chosen server does not have is an error at start, not silently ignored. `docker run ... kyojin shell` opens a shell and `... kyojin check` runs `tools/strix_halo/env.sh --check`. To see the command a configuration produces without a GPU: `docker run --rm -e KYOJIN_MODEL=glm -v ...:/models:ro --entrypoint python kyojin docker/kyojin_args.py`.

## Compose, behind llama-swap

Not tested: the compose file and the llama-swap entry below were written from the tested `docker run` line above.

```yaml
services:
  kyojin-qwen:
    image: kyojin
    devices: ["/dev/kfd:/dev/kfd", "/dev/dri:/dev/dri"]
    group_add: ["${VIDEO_GID}", "${RENDER_GID}"]   # numeric ids from: getent group video render
    security_opt: ["seccomp=unconfined"]
    ipc: host
    volumes: ["/path/to/Qwen3.8-Flash-Next-EXL3-Yamz:/models:ro"]
    environment:
      KYOJIN_MODEL: qwen
      CONTEXT: "65536"
      KYOJIN_MODEL_ID: qwen38
    ports: ["8101:8000"]
    stop_grace_period: 30s
    restart: "no"
```

llama-swap config entry (it starts and stops the container itself; `cmd` runs in the foreground, `cmdStop` is optional because llama-swap sends SIGTERM to the `docker run` process, which forwards it):

```yaml
models:
  qwen38:
    cmd: >
      docker run --rm --name kyojin-qwen --device /dev/kfd --device /dev/dri --group-add ${VIDEO_GID} --group-add ${RENDER_GID}
      --security-opt seccomp=unconfined --ipc=host -v /path/to/pack:/models:ro
      -e KYOJIN_MODEL=qwen -e CONTEXT=65536 -p 8101:8000 kyojin
    cmdStop: docker stop kyojin-qwen
    proxy: http://127.0.0.1:8101
    checkEndpoint: /health
    ttl: 600
```

`/health` answers 200 once the port is open, which is after the model is loaded and warmed up. Docker's own `HEALTHCHECK` uses the same endpoint, with a 15-minute start period.

## Stopping

The entrypoint runs the server as a child and forwards SIGTERM to it. Measured on the Qwen pack: `docker stop` returns in about a second, exit code 143, the same as SIGTERM on a bare-metal server. A GPU-memory-heavy server can take some seconds to free the device; keep `stop_grace_period` at 30 s or more.

## Notes

- One container, one model, one GPU. Two containers on the same GPU will not both fit in 128 GB.
- The image holds no model and no secrets. `hf download` the pack on the host first.
- `--ipc=host` and `seccomp=unconfined` are the usual ROCm container flags; test without them on your host if you prefer a tighter setup.

#!/bin/bash
# Container entrypoint. Every setting is an optional environment variable (see doc/container.md): the image knows
# its model (KYOJIN_MODEL), its defaults for a 128 GB machine, and it downloads a missing pack into /models.
# Arguments given to `docker run IMAGE <args>` are appended to the server command line.
set -u
ROOT=${KYOJIN_ROOT:-/opt/kyojin}
cd "$ROOT"
source tools/strix_halo/env.sh >/dev/null

case "${1:-}" in
    shell) shift; exec bash "$@" ;;
    check) exec bash tools/strix_halo/env.sh --check ;;
esac

# Kernel cache (compiled HIP kernels, tuning tables, Triton): ~/.cache is empty in a fresh container and the first
# start then takes 4 more minutes. Keep it across runs: a volume mounted on ~/.cache wins, else a hidden folder in
# the writable models mount, else the container's own (lost with it). KYOJIN_CACHE_DIR forces a folder.
CACHE_HOME=${HOME:-/root}/.cache
CACHE_DIR=${KYOJIN_CACHE_DIR:-}
if [ -z "$CACHE_DIR" ] && ! mountpoint -q "$CACHE_HOME" 2>/dev/null; then
    MODELS=${KYOJIN_MODEL_DIR:-${MODEL_DIR:-/models}}
    [ -d "$MODELS" ] && [ -w "$MODELS" ] && CACHE_DIR=$MODELS/.kyojin-cache
fi
if [ -n "$CACHE_DIR" ] && [ "$CACHE_DIR" != "$CACHE_HOME" ]; then
    if mkdir -p "$CACHE_DIR" 2>/dev/null && { [ -L "$CACHE_HOME" ] || rmdir "$CACHE_HOME" 2>/dev/null || [ ! -e "$CACHE_HOME" ]; } \
        && ln -sfn "$CACHE_DIR" "$CACHE_HOME"; then
        echo "[kyojin] kernel cache: $CACHE_DIR" >&2
    else
        echo "[kyojin] kernel cache: cannot use $CACHE_DIR, using $CACHE_HOME (lost when the container is removed)" >&2
    fi
else
    echo "[kyojin] kernel cache: $CACHE_HOME" >&2
fi
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$CACHE_HOME/triton}

# Children run in the background so that this shell (PID 1) can forward SIGTERM: a bare PID 1 ignores it and docker
# would wait for SIGKILL. SIGTERM, not SIGINT: a background job of a non-interactive shell starts with SIGINT ignored,
# so python never sees it (measured: docker stop took the full timeout, exit 137). The trap is set before the
# configuration step because that step can be a download of tens of GB.
CHILD=""
stop() { echo "[kyojin] signal received, stopping" >&2; [ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null; STOPPED=1; }
STOPPED=0
trap stop TERM INT
wait_child() { while kill -0 "$CHILD" 2>/dev/null; do wait "$CHILD" 2>/dev/null; done; wait "$CHILD" 2>/dev/null; return $?; }

ARGFILE=$(mktemp)
python docker/kyojin_args.py --null >"$ARGFILE" &
CHILD=$!
wait_child; RC=$?
if [ "$RC" -ne 0 ] || [ "$STOPPED" = 1 ]; then rm -f "$ARGFILE"; [ "$STOPPED" = 1 ] && exit 143; exit "$RC"; fi
mapfile -d '' ARGS <"$ARGFILE"; rm -f "$ARGFILE"
[ "${#ARGS[@]}" -gt 0 ] || exit 2
echo "[kyojin] python ${ARGS[*]} $*" >&2

python "${ARGS[@]}" "$@" &
CHILD=$!
wait_child; RC=$?
echo "[kyojin] server exited with code $RC" >&2
exit "$RC"

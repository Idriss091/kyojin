#!/bin/bash
# Container entrypoint: configuration comes from environment variables (see doc/container.md).
# Arguments given to `docker run IMAGE <args>` are appended to the server command line.
set -u
cd /opt/kyojin
source tools/strix_halo/env.sh >/dev/null

case "${1:-}" in
    shell) shift; exec bash "$@" ;;
    check) exec bash tools/strix_halo/env.sh --check ;;
esac

mapfile -d '' ARGS < <(python docker/kyojin_args.py --null) || true
[ "${#ARGS[@]}" -gt 0 ] || exit 2
echo "[kyojin] python ${ARGS[*]} $*" >&2

# Run the server as a child so this shell (PID 1) can forward SIGTERM. A bare PID 1 ignores SIGTERM and docker
# would wait for SIGKILL. Forward SIGTERM, not SIGINT: a background job of a non-interactive shell starts with SIGINT
# ignored, so python never sees it (measured: docker stop took the full timeout and ended in exit 137).
python "${ARGS[@]}" "$@" &
PID=$!
stop() { echo "[kyojin] signal received, stopping server" >&2; kill -TERM "$PID" 2>/dev/null; }
trap stop TERM INT
while kill -0 "$PID" 2>/dev/null; do wait "$PID" 2>/dev/null; done
wait "$PID" 2>/dev/null; RC=$?
echo "[kyojin] server exited with code $RC" >&2
exit "$RC"

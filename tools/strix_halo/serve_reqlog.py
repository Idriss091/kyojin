#!/usr/bin/env python3
"""Per-request state log for the Kyojin servers: timings and counters only, never text.

A request that is refused, queued or prefilling produces **no client-visible output**, so a report of
"the server went silent" cannot be lined up with what the server was doing. This writes one line per
lifecycle event, plus a heartbeat while a job is queued or running, so the two timelines can be
compared afterwards.

**Off unless asked for.** With KYOJIN_REQLOG unset (or 0/off/none/no/false/empty) nothing is opened and
every call is a no-op, so a default deployment behaves exactly as before. The server prints one line at
start-up either way, so the switch stays discoverable when a request does go silent.

    KYOJIN_REQLOG=~/cache/kyojin/requests.log   # a path, or 1/on/yes for the default path below
    KYOJIN_REQLOG_HEARTBEAT_S=10                # seconds between 'alive' lines while a job runs

Privacy: nothing derived from the request is written except numbers — token counts, message/image
counts, durations, byte counts, HTTP statuses, an opaque request id, a server tag and event names.
Prompt text, completion text, tool arguments and image paths from the body are never logged. Of the
server's own error text, only the exception type is logged plus a truncated message; JSON parse errors
are reduced to line/column because their message can quote the request body.

Reading it back: `grep req_refused`, `grep img_cache`, `grep -E 'alive|job_first_token'`, or follow one
request with `grep <req id>`. Every line starts with a local ISO timestamp with milliseconds, then the
event name, then `key=value` fields; every line of one request carries the same `req=` id.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import threading
import time
from typing import Any

DEFAULT_PATH = "~/.cache/kyojin/requests.log"
FALSE = {"0", "off", "none", "no", "false", ""}
TRUE = {"1", "on", "yes", "true"}


def sanitize(exc: BaseException) -> str:
    """Exception type and a short message, safe to log (no request-body snippets)."""
    if isinstance(exc, json.JSONDecodeError):
        return f"JSONDecodeError(line={exc.lineno} col={exc.colno})"
    msg = " ".join(str(exc).split())[:160]
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        # Whole numbers read as integers (cache_pct=40, budget_mb=1024), the rest keep milliseconds.
        return str(int(value)) if value.is_integer() else f"{value:.3f}"
    text = str(value)
    return f'"{text}"' if (" " in text or "=" in text) else text


class ReqLog:
    """Append-only event log. Every method is safe to call when logging is disabled or broken."""

    def __init__(self, path: str | None = None, heartbeat_s: float | None = None,
                 server: str | None = None) -> None:
        raw = os.environ.get("KYOJIN_REQLOG", "") if path is None else path
        raw = raw.strip()
        self.server = server
        self.enabled = raw.lower() not in FALSE
        if self.enabled and raw.lower() in TRUE:
            raw = DEFAULT_PATH
        self.path: pathlib.Path | None = pathlib.Path(raw).expanduser() if self.enabled else None
        if heartbeat_s is None:
            try:
                heartbeat_s = float(os.environ.get("KYOJIN_REQLOG_HEARTBEAT_S", "10") or 10)
            except ValueError:
                heartbeat_s = 10.0
        self.heartbeat_s = max(0.0, heartbeat_s)
        self._lock = threading.Lock()
        self._fh = None
        if self.enabled and self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = self.path.open("a", encoding="utf-8")
            except OSError as exc:
                print(f"serve: request log off ({type(exc).__name__}: {exc})", file=sys.stderr, flush=True)
                self.enabled, self._fh = False, None

    @classmethod
    def from_env(cls, server: str | None = None) -> "ReqLog":
        return cls(server=server)

    def describe(self) -> str:
        """One start-up line, so an operator can tell whether the log is on without looking for a file."""
        if self._fh is None or self.path is None:
            return "request log off (set KYOJIN_REQLOG=<path> to enable, numbers and event names only)"
        return (f"request log {self.path} (heartbeat {self.heartbeat_s:g} s, "
                f"numbers and event names only, no prompt or reply text)")

    def log(self, event: str, **fields: Any) -> None:
        if self._fh is None:
            return
        try:
            stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
            ms = int((time.time() % 1) * 1000)
            parts = [f"{stamp}.{ms:03d}", event]
            if self.server:
                parts.append(f"srv={self.server}")
            for key, value in fields.items():
                if value is None:
                    continue
                parts.append(f"{key}={_fmt(value)}")
            line = " ".join(parts) + "\n"
            with self._lock:
                self._fh.write(line)
                self._fh.flush()
        except Exception:  # noqa: BLE001  a logging failure must never touch serving
            pass

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                except OSError:
                    pass
                self._fh = None

    def lines(self) -> list[str]:
        """The lines written so far (for tests and for a live tail; never opens a second handle)."""
        if self.path is None or not self.path.exists():
            return []
        with self._lock:
            if self._fh is not None:
                self._fh.flush()
        return self.path.read_text(encoding="utf-8").splitlines()

"""GPU locks: one holder per job, flock waits, released between jobs. Never blocks another worker longer than one job."""
from __future__ import annotations

import contextlib
import fcntl
import os
import time

MIN_MEM_GIB = 4.0


@contextlib.contextmanager
def gpu_locks(paths: list[str], log=print, wait_s: float = 6 * 3600):
    """Take every lock in order (flock, polling so a wait can be reported); release in reverse order."""
    fds = []
    try:
        for p in paths:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            fd = os.open(p, os.O_RDWR | os.O_CREAT, 0o644)
            t0 = time.time(); said = False
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); break
                except BlockingIOError:
                    if time.time() - t0 > wait_s:
                        os.close(fd); raise TimeoutError(f"lock {p} not free after {wait_s:.0f} s")
                    if not said:
                        log(f"waiting for lock {p}"); said = True
                    time.sleep(5)
            fds.append(fd)
            log(f"lock held {p} (waited {time.time()-t0:.0f} s)")
        yield
    finally:
        for fd in reversed(fds):
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd)
        log("locks released")

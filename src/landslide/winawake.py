"""Keep Windows awake while the overnight run works (BRIEF.md Phase 1 item 6 "Awake"; PLAN.md P1-14 item 3).

`keep_awake()` calls `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` through ctypes on entry and
`SetThreadExecutionState(ES_CONTINUOUS)` on exit (in `finally`), which clears the request. Nothing else changes: no
power plan, sleep timeout or display setting is touched, and the display may still turn off. The request belongs to
the calling thread, so the runner enters the context on its main thread, which does all the work. Off Windows the
context is a no-op.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from landslide.logs import get_logger

log = get_logger("winawake")

#: winbase.h execution-state flags.
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
AWAKE_FLAGS = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
RELEASE_FLAGS = ES_CONTINUOUS


def _kernel32_setter() -> Callable[[int], int]:
    """`kernel32.SetThreadExecutionState` with its C signature (DWORD -> DWORD; 0 means failure)."""
    import ctypes
    from ctypes import wintypes

    fn = ctypes.WinDLL("kernel32", use_last_error=True).SetThreadExecutionState
    fn.argtypes = [wintypes.DWORD]
    fn.restype = wintypes.DWORD
    return fn


@contextmanager
def keep_awake(setter: Callable[[int], int] | None = None, platform: str | None = None) -> Iterator[bool]:
    """Ask Windows not to sleep until the block ends. Yields True if the request is active.

    `setter` replaces `SetThreadExecutionState` and `platform` replaces `sys.platform` (tests only). A failed call
    (return value 0) is logged and the run goes on: sleeping is a risk, not a reason to stop.
    """
    plat = sys.platform if platform is None else platform
    if plat != "win32":
        log.info("keep-awake: not Windows (%s), nothing to do", plat)
        yield False
        return
    fn = setter if setter is not None else _kernel32_setter()
    prev = fn(AWAKE_FLAGS)
    active = bool(prev)
    if active:
        log.info("keep-awake: SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) set; power settings "
                 "unchanged")
    else:
        log.warning("keep-awake: SetThreadExecutionState failed; Windows may sleep during the run")
    try:
        yield active
    finally:
        if active:
            fn(RELEASE_FLAGS)
            log.info("keep-awake: released (ES_CONTINUOUS)")

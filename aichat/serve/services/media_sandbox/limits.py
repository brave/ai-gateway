from __future__ import annotations

import logging
import resource
import signal
import sys

logger = logging.getLogger(__name__)

CPU_SECONDS = 5
MEMORY_BYTES = 768 * 1024 * 1024
FILE_SIZE_BYTES = 64 * 1024 * 1024
NOFILE = 64

# Single source of truth for op timeout budgets: the in-worker alarm must
# fire strictly before the parent's hard kill so the worker can report a
# clean error instead of dying silently.
IN_WORKER_WALL_CLOCK_SECONDS = 13
PARENT_HARD_KILL_SECONDS = 15


class OpWallClockExceeded(Exception):
    pass


def _on_sigalrm(signum, frame) -> None:
    raise OpWallClockExceeded("op exceeded in-worker wall clock")


def _lower_rlimit(res, value, linux: bool, name: str) -> None:
    try:
        _, hard = resource.getrlimit(res)
        if hard != resource.RLIM_INFINITY:
            value = min(value, hard)
        resource.setrlimit(res, (value, value))
    except (ValueError, OSError):
        if linux:
            raise
        logger.warning("rlimit %s not enforceable on this platform", name)


def apply_rlimits(
    cpu_seconds: int = CPU_SECONDS,
    memory_bytes: int = MEMORY_BYTES,
    file_size_bytes: int = FILE_SIZE_BYTES,
    nofile: int = NOFILE,
) -> None:
    linux = sys.platform.startswith("linux")

    _lower_rlimit(resource.RLIMIT_AS, memory_bytes, linux, "AS")
    _lower_rlimit(resource.RLIMIT_FSIZE, file_size_bytes, linux, "FSIZE")
    _lower_rlimit(resource.RLIMIT_NOFILE, nofile, linux, "NOFILE")

    try:
        _, hard = resource.getrlimit(resource.RLIMIT_CPU)
        soft = cpu_seconds
        if hard != resource.RLIM_INFINITY and soft > hard:
            soft = hard
        resource.setrlimit(resource.RLIMIT_CPU, (soft, hard))
    except (ValueError, OSError):
        if linux:
            raise
        logger.warning("rlimit CPU not enforceable on this platform")

    signal.signal(signal.SIGXCPU, signal.SIG_DFL)


def arm_op_limits(
    cpu_seconds: int = CPU_SECONDS,
    wall_clock_seconds: int = IN_WORKER_WALL_CLOCK_SECONDS,
) -> None:
    used = resource.getrusage(resource.RUSAGE_SELF)
    consumed = used.ru_utime + used.ru_stime
    next_soft = int(consumed + cpu_seconds) + 1
    try:
        _, hard = resource.getrlimit(resource.RLIMIT_CPU)
        if hard != resource.RLIM_INFINITY and next_soft > hard:
            next_soft = hard
        resource.setrlimit(resource.RLIMIT_CPU, (next_soft, hard))
    except (ValueError, OSError):
        if sys.platform.startswith("linux"):
            raise
        logger.warning("rlimit CPU refresh not enforceable on this platform")

    signal.signal(signal.SIGALRM, _on_sigalrm)
    signal.alarm(wall_clock_seconds)


def disarm_op_limits() -> None:
    signal.alarm(0)

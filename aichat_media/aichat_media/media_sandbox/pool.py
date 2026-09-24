from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass

from aichat_media.media_sandbox import protocol
from aichat_media.media_sandbox.limits import PARENT_HARD_KILL_SECONDS

logger = logging.getLogger(__name__)

WORKER_MODULE = "aichat_media.media_sandbox.worker"
WALL_CLOCK_SECONDS = PARENT_HARD_KILL_SECONDS
POOL_SIZE = 2
SPAWN_TIMEOUT = 10.0
# JSON-lines payloads carry base64 media (e.g. decoded PCM); allow up to 128MB lines.
STREAM_LIMIT_BYTES = 128 * 1024 * 1024


class SandboxWorkerError(RuntimeError):
    pass


class SandboxOpError(SandboxWorkerError):
    pass


@dataclass
class _Pending:
    fut: asyncio.Future


class _Worker:
    def __init__(self) -> None:
        self.proc: asyncio.subprocess.Process | None = None
        self.pending: dict[int, _Pending] = {}
        self._next_id = 0
        self.alive = False
        self._start_lock = asyncio.Lock()

    async def start(self) -> None:
        self.proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            WORKER_MODULE,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,
            limit=STREAM_LIMIT_BYTES,
        )
        asyncio.get_running_loop().create_task(self._read_loop())
        try:
            resp = await self._roundtrip("ping", {}, timeout=SPAWN_TIMEOUT)
        except (TimeoutError, SandboxWorkerError):
            await self.kill()
            raise SandboxWorkerError(
                "worker startup failed (ping timeout/death)"
            ) from None
        if resp != {"pong": True}:
            await self.kill()
            raise SandboxWorkerError(f"worker ping failed: {resp}")
        self.alive = True

    async def _roundtrip(self, op: str, args: dict, timeout: float) -> dict:
        req = protocol.Request(id=self._next_id, op=op, args=args)
        self._next_id += 1
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self.pending[req.id] = _Pending(fut=fut)
        try:
            assert self.proc and self.proc.stdin
            self.proc.stdin.write(req.encode())
            await self.proc.stdin.drain()
            return await asyncio.wait_for(fut, timeout)
        finally:
            self.pending.pop(req.id, None)

    async def call(self, op: str, args: dict) -> dict:
        if not self.alive:
            # Serialize (re)starts so concurrent callers cannot double-start
            # the same dead worker.
            async with self._start_lock:
                if not self.alive:
                    try:
                        await self.start()
                    except Exception as e:
                        raise SandboxWorkerError(
                            f"failed to restart sandbox worker: {e}"
                        ) from e
        try:
            return await self._roundtrip(op, args, timeout=WALL_CLOCK_SECONDS)
        except TimeoutError:
            await self.kill()
            raise SandboxWorkerError(
                f"media op '{op}' exceeded {WALL_CLOCK_SECONDS}s wall clock; "
                "worker killed"
            ) from None
        except (ConnectionResetError, BrokenPipeError) as e:
            self.alive = False
            raise SandboxWorkerError(
                f"worker died during op '{op}' "
                f"(likely rlimit/OOM/Landlock kill): {e}"
            ) from e

    async def _read_loop(self) -> None:
        assert self.proc and self.proc.stdout
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                break
            try:
                resp = protocol.Response.decode(line)
            except protocol.ProtocolError:
                logger.warning("media worker sent malformed response: %r", line)
                continue
            pending = self.pending.pop(resp.id, None)
            if pending and not pending.fut.done():
                if resp.ok:
                    pending.fut.set_result(resp.value)
                else:
                    pending.fut.set_exception(
                        SandboxOpError(
                            f"op failed in sandbox: {resp.exc_type}: {resp.error}"
                        )
                    )
        self.alive = False
        rc = await self.proc.wait() if self.proc else 0
        err = None
        if rc != 0:
            err = f"worker exited rc={rc}"
        for p in self.pending.values():
            if not p.fut.done():
                p.fut.set_exception(
                    SandboxWorkerError(f"worker exited (rc={rc}): {err or 'EOF'}")
                )
        self.pending.clear()

    async def kill(self) -> None:
        self.alive = False
        if self.proc and self.proc.returncode is None:
            try:
                self.proc.kill()
            except ProcessLookupError:
                pass
            await self.proc.wait()


class MediaSandboxPool:
    def __init__(self, size: int = POOL_SIZE):
        self._size = size
        self._workers: list[_Worker] = []
        self._lock = asyncio.Lock()
        self._rr = 0
        self._started = False
        self._start_error: Exception | None = None

    async def _ensure_workers(self) -> None:
        if self._started:
            return
        async with self._lock:
            if self._started:
                return
            if self._start_error:
                raise self._start_error
            try:
                for _ in range(self._size):
                    w = _Worker()
                    await w.start()
                    self._workers.append(w)
            except Exception as e:
                self._start_error = SandboxWorkerError(
                    f"sandbox workers failed to start: {e}"
                )
                raise self._start_error from e
            self._started = True

    async def call(self, op: str, args: dict) -> dict:
        await self._ensure_workers()
        for _ in range(len(self._workers)):
            w = self._workers[self._rr % len(self._workers)]
            self._rr += 1
            try:
                return await w.call(op, args)
            except SandboxOpError:
                raise
            except SandboxWorkerError:
                continue
        raise SandboxWorkerError(f"all {len(self._workers)} sandbox workers failed")

    async def shutdown(self) -> None:
        for w in self._workers:
            await w.kill()
        self._workers.clear()
        self._started = False
        self._start_error = None
        self._rr = 0


_pool: MediaSandboxPool | None = None


def get_pool() -> MediaSandboxPool:
    global _pool
    if _pool is None:
        _pool = MediaSandboxPool()
    return _pool


async def shutdown_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.shutdown()
        _pool = None


def reset_pool() -> None:
    global _pool
    if _pool is not None:
        for w in _pool._workers:
            if w.proc is not None and w.proc.returncode is None:
                try:
                    w.proc.kill()
                except ProcessLookupError:
                    pass
                # Reap so the child doesn't linger as a zombie when called
                # from a running event loop (tests wrap this in asyncio.run).
                try:
                    t = asyncio.get_running_loop().create_task(w.proc.wait())
                except RuntimeError:
                    pass
                else:
                    _reap_tasks.add(t)
                    t.add_done_callback(_reap_tasks.discard)
    _pool = None


_reap_tasks: set[asyncio.Task] = set()

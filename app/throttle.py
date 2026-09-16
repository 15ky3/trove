"""Bandwidth limiting for transfers.

The bytes of a download never pass through Python: huggingface_hub hands the
work to hf_xet, which transfers in Rust, and the tqdm objects the worker
patches only report what has already arrived. Sleeping there would slow the
readout, not the line.

The one chokepoint both the Rust client and the Python HTTP path share is the
proxy they read from the environment, so that is where the limit goes: a proxy
of our own, bound to localhost, counting the bytes it relays.

Only CONNECT is served. Every Hub endpoint is https and a CONNECT tunnel is
relayed byte for byte — no TLS is terminated, no certificate has to exist, and
a repo that is not Xet-backed travels through the same tunnel.

The kernel would be the better place for this, but the one this runs on has
neither an ingress qdisc nor ifb, so shaping a container's *incoming* traffic
is not available. Userspace it is.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Any, Callable

#: Mbit/s as networks mean it: 10^6 bits, not 2^20.
BYTES_PER_MBIT = 1_000_000 / 8

#: Bytes moved per relay step. Large enough to keep the syscall count sane,
#: small enough that the limit is smooth rather than a burst every second.
CHUNK = 64 * 1024

#: How much traffic may accumulate while nothing is being fetched. A quarter of
#: a second keeps the average honest without stalling the first read.
BURST_SECONDS = 0.25
MIN_BURST = CHUNK

#: Never sleep for less than this; a shorter wait only burns CPU.
MIN_SLEEP = 0.005

#: Longest wait for the request headers, and for the upstream connection.
CONNECT_TIMEOUT = 20.0

#: Request line plus headers. A CONNECT request is a single short line; anything
#: larger is not a client we serve.
HEADER_LIMIT = 32 * 1024

#: Every spelling a client might read. httpx and reqwest both look at the
#: uppercase and the lowercase form.
PROXY_VARS = (
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
)


class TokenBucket:
    """Hands out bytes at a fixed rate.

    Pure arithmetic with an injectable clock: `grant` never sleeps, it says how
    much may go now and how long to wait when the answer is nothing. That keeps
    the maths testable and leaves the waiting to the caller, which knows whether
    it is in a thread or on an event loop.
    """

    def __init__(
        self,
        rate: float,
        burst: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = float(rate)
        self.capacity = float(burst) if burst else max(self.rate * BURST_SECONDS, MIN_BURST)
        self._clock = clock
        self._tokens = self.capacity
        self._last = clock()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = self._clock()
        earned = max(0.0, now - self._last) * self.rate
        self._tokens = min(self.capacity, self._tokens + earned)
        self._last = now

    def grant(self, want: int) -> tuple[int, float]:
        """Return (bytes allowed now, seconds to wait when that is zero)."""
        if want <= 0:
            return 0, 0.0
        with self._lock:
            self._refill()
            if self._tokens >= 1:
                granted = int(min(want, self._tokens))
                self._tokens -= granted
                return granted, 0.0
            # Wait for a whole chunk rather than for a single byte, or the
            # caller wakes up thousands of times a second to move nothing.
            deficit = min(want, self.capacity) - self._tokens
            return 0, max(deficit / self.rate, MIN_SLEEP)

    def refund(self, amount: int) -> None:
        """Give back tokens taken for bytes that never arrived."""
        if amount <= 0:
            return
        with self._lock:
            self._tokens = min(self.capacity, self._tokens + amount)


def _close(writer: asyncio.StreamWriter) -> None:
    try:
        writer.close()
    except Exception:  # noqa: BLE001 - closing a dead socket is not news
        pass


async def _reply(writer: asyncio.StreamWriter, status: int, reason: str) -> None:
    body = reason.encode("utf-8", "replace")
    head = (
        f"HTTP/1.1 {status} {reason}\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Content-Type: text/plain\r\n"
        "Connection: close\r\n\r\n"
    ).encode("latin-1")
    try:
        writer.write(head + body)
        await writer.drain()
    except (OSError, ConnectionError):
        pass
    _close(writer)


class ThrottledProxy:
    """A CONNECT proxy on localhost that caps what it relays downstream.

    Upstream (what we send to the Hub) runs unthrottled: this exists to stop a
    download from eating the line, and an upload is paced by the same setting
    only if someone asks for it.
    """

    def __init__(self, rate: float, host: str = "127.0.0.1") -> None:
        self.bucket = TokenBucket(rate)
        self.host = host
        self.url = ""
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopped: asyncio.Event | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None

    # ------------------------------------------------------------- lifecycle

    def start(self) -> str:
        """Bring the proxy up and return its URL. Raises if it cannot bind."""
        self._thread = threading.Thread(target=self._serve, name="throttle", daemon=True)
        self._thread.start()
        if not self._ready.wait(10):
            raise RuntimeError("the speed limiter did not start in time")
        if self._error is not None:
            raise self._error
        return self.url

    def stop(self) -> None:
        loop = self._loop
        stopped = self._stopped
        if loop is not None and stopped is not None and not loop.is_closed():
            loop.call_soon_threadsafe(stopped.set)
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _serve(self) -> None:
        try:
            asyncio.run(self._main())
        except BaseException as exc:  # noqa: BLE001 - reported to start()
            self._error = exc
        finally:
            self._ready.set()

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stopped = asyncio.Event()
        server = await asyncio.start_server(
            self._client, self.host, 0, limit=HEADER_LIMIT
        )
        self.url = f"http://{self.host}:{server.sockets[0].getsockname()[1]}"
        self._ready.set()
        async with server:
            await self._stopped.wait()

    # --------------------------------------------------------------- serving

    async def _client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), CONNECT_TIMEOUT)
        except asyncio.LimitOverrunError:
            await _reply(writer, 400, "Request headers too large")
            return
        except (asyncio.IncompleteReadError, asyncio.TimeoutError, OSError, ConnectionError):
            _close(writer)
            return

        target = _connect_target(head)
        if target is None:
            await _reply(writer, 400, "Malformed request")
            return
        if target == ("", 0):
            await _reply(writer, 405, "Only CONNECT is proxied")
            return

        host, port = target
        try:
            up_reader, up_writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), CONNECT_TIMEOUT
            )
        except (OSError, ConnectionError, asyncio.TimeoutError):
            await _reply(writer, 502, f"Cannot reach {host}:{port}")
            return

        try:
            writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            await writer.drain()
        except (OSError, ConnectionError):
            _close(writer)
            _close(up_writer)
            return

        await asyncio.gather(
            self._relay(up_reader, writer, self.bucket),
            self._relay(reader, up_writer, None),
            return_exceptions=True,
        )
        _close(writer)
        _close(up_writer)

    async def _relay(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        bucket: TokenBucket | None,
    ) -> None:
        try:
            while True:
                size = CHUNK if bucket is None else await self._take(bucket, CHUNK)
                data = await reader.read(size)
                if not data:
                    break
                if bucket is not None and len(data) < size:
                    # Paid for more than the socket had; the rest stays ours.
                    bucket.refund(size - len(data))
                writer.write(data)
                await writer.drain()
        except (OSError, ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            # The peer direction blocks on a read that will never return once
            # this side is done, so close it rather than leave it hanging.
            _close(writer)

    @staticmethod
    async def _take(bucket: TokenBucket, want: int) -> int:
        while True:
            granted, wait = bucket.grant(want)
            if granted:
                return granted
            await asyncio.sleep(wait)


def _connect_target(head: bytes) -> tuple[str, int] | None:
    """Parse the request line.

    Returns the host and port for a CONNECT, `("", 0)` for any other method, and
    None when the line is not a request line at all.
    """
    line = head.split(b"\r\n", 1)[0].decode("latin-1", "replace")
    parts = line.split()
    if len(parts) < 2:
        return None
    if parts[0].upper() != "CONNECT":
        return "", 0
    host, _, port = parts[1].rpartition(":")
    if not host or not port.isdigit():
        return None
    number = int(port)
    if not 0 < number < 65536:
        return None
    return host.strip("[]"), number


def start_limit(mbit: Any, log: Callable[..., None] | None = None) -> ThrottledProxy | None:
    """Start a limiter for this process and point the proxy variables at it.

    Zero, nothing, or a value that is not a number means no limit and no proxy —
    the transfer then runs exactly as it did before this module existed. A
    limiter that cannot bind is reported and skipped: a speed setting is not
    worth failing a download over.
    """
    try:
        rate = float(mbit or 0)
    except (TypeError, ValueError):
        rate = 0.0
    if rate <= 0:
        return None

    proxy = ThrottledProxy(rate * BYTES_PER_MBIT)
    try:
        url = proxy.start()
    except Exception as exc:  # noqa: BLE001 - never fatal
        if log is not None:
            log(f"Speed limit of {rate:g} Mbit/s not applied: {exc}", "warn")
        return None

    for var in PROXY_VARS:
        os.environ[var] = url
    if log is not None:
        log(f"Speed limit: {rate:g} Mbit/s.")
    return proxy

"""Bandwidth limiting for transfers.

The bytes of a download never pass through Python: huggingface_hub hands the
work to hf_xet, which transfers in Rust, and the tqdm objects the worker
patches only report what has already arrived. Sleeping there would slow the
readout, not the line.

The one chokepoint both the Rust client and the Python HTTP path share is the
proxy they read from the environment, so that is where the limit goes: a proxy
of our own, bound to localhost, counting the bytes it relays.

One proxy serves every worker, and it lives in the API process rather than in
each of them. A ceiling that each transfer applies for itself is not a ceiling:
two parallel downloads would take twice the configured rate. Shared, the number
in Settings is the number on the line, and changing it retunes the transfers
that are already running.

Only CONNECT is served. Every Hub endpoint is https and a CONNECT tunnel is
relayed byte for byte — no TLS is terminated, no certificate has to exist, and
a repo that is not Xet-backed travels through the same tunnel.

The kernel would be the better place for this, but the one this runs on has
neither an ingress qdisc nor ifb, so shaping a container's *incoming* traffic
is not available. Userspace it is.
"""

from __future__ import annotations

import asyncio
import base64
import os
import threading
import time
from typing import Any, Callable
from urllib.parse import urlsplit

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

#: How long to wait for the proxy thread to report a bound port.
START_TIMEOUT = 10.0

#: What a worker is told to use. Only the https spellings: this proxy speaks
#: CONNECT and nothing else, and a plain-http endpoint — a self-hosted mirror
#: is allowed to be one — would be sent here as an absolute-form GET and
#: refused. Such a mirror now goes out directly, unthrottled but working.
#: Measured: the Xet client routes through HTTPS_PROXY alone, ALL_PROXY is not
#: needed. Both cases are covered because httpx and reqwest disagree on which
#: spelling they read.
PROXY_VARS = ("HTTPS_PROXY", "https_proxy")

#: Cleared while a limit is on. An operator who excluded the Hub from their
#: company proxy would otherwise send the worker straight past the limiter,
#: with the interface still showing a ceiling that does nothing.
NO_PROXY_VARS = ("NO_PROXY", "no_proxy")


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

    def set_rate(self, rate: float) -> None:
        """Retune while running. Tokens already earned survive, up to the new
        burst size — a ceiling lowered mid-transfer must not stay generous."""
        if rate <= 0:
            raise ValueError("rate must be positive")
        with self._lock:
            self._refill()
            self.rate = float(rate)
            self.capacity = max(self.rate * BURST_SECONDS, MIN_BURST)
            self._tokens = min(self._tokens, self.capacity)


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

    def __init__(
        self,
        rate: float,
        host: str = "127.0.0.1",
        upstream_proxy: str = "",
    ) -> None:
        #: None means everything goes through untouched. A proxy without a
        #: ceiling still has to exist: workers pointed at it keep using it for
        #: their whole life, and pulling it away mid-transfer kills them.
        self.bucket: TokenBucket | None = TokenBucket(rate)
        self.host = host
        #: A proxy the operator configured for the container. In a network with
        #: no direct way out, dialling the Hub ourselves would turn every
        #: download into a 502 as soon as a limit is set, so we go through
        #: theirs and meter what comes back.
        self.upstream_proxy = upstream_proxy
        self.url = ""
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopped: asyncio.Event | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._abandoned = False
        #: Tunnels currently being served, so stop() can end them.
        self._clients: set[asyncio.Task] = set()

    def set_rate(self, rate: float | None) -> None:
        """Retune, or let everything through when given None.

        Open tunnels read the bucket on every chunk, so a change reaches the
        transfers that are running, not just the next ones.
        """
        if rate is None:
            self.bucket = None
        elif self.bucket is None:
            self.bucket = TokenBucket(rate)
        else:
            self.bucket.set_rate(rate)

    # ------------------------------------------------------------- lifecycle

    def start(self) -> str:
        """Bring the proxy up and return its URL. Raises if it cannot bind."""
        self._thread = threading.Thread(target=self._serve, name="throttle", daemon=True)
        self._thread.start()
        if not self._ready.wait(START_TIMEOUT):
            # Whatever it is doing, nobody will ever use it. Say so, so that a
            # thread which binds a moment later takes itself down again instead
            # of holding a port for the life of the process.
            self._abandoned = True
            self.stop()
            raise RuntimeError("the speed limiter did not start in time")
        if self._error is not None:
            raise self._error
        return self.url

    def stop(self) -> None:
        loop = self._loop
        stopped = self._stopped
        if loop is not None and stopped is not None and not loop.is_closed():
            loop.call_soon_threadsafe(stopped.set)
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5)
            # Only let go of a thread that actually ended. Dropping the handle
            # of one that did not is how a proxy becomes unreachable and
            # unstoppable while still holding its port.
            if not thread.is_alive():
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
        if self._abandoned:
            self._stopped.set()
        try:
            await self._stopped.wait()
        finally:
            server.close()
            # Open tunnels have to be ended by hand. Waiting for the server to
            # close waits for its handlers too (3.12.1 and newer), so a caller
            # in stop() would sit here for as long as a transfer runs — and
            # stop() is called from the request that clears the setting, which
            # means the whole ASGI loop would sit here with it.
            for task in list(self._clients):
                task.cancel()
            if self._clients:
                await asyncio.wait(self._clients, timeout=2)
            await server.wait_closed()

    # --------------------------------------------------------------- serving

    async def _client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._clients.add(task)
        try:
            await self._tunnel(reader, writer)
        except asyncio.CancelledError:
            _close(writer)
        finally:
            self._clients.discard(task)

    async def _tunnel(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
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
                self._open_upstream(host, port), CONNECT_TIMEOUT
            )
        except (OSError, ConnectionError, asyncio.TimeoutError, asyncio.IncompleteReadError):
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
            self._relay(up_reader, writer, throttled=True),
            self._relay(reader, up_writer, throttled=False),
            return_exceptions=True,
        )
        _close(writer)
        _close(up_writer)

    async def _open_upstream(
        self, host: str, port: int
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Reach the target, through the operator's proxy when there is one."""
        if not self.upstream_proxy:
            return await asyncio.open_connection(host, port)

        parts = urlsplit(self.upstream_proxy)
        reader, writer = await asyncio.open_connection(parts.hostname, parts.port or 8080)
        request = f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n"
        if parts.username:
            secret = f"{parts.username}:{parts.password or ''}".encode()
            request += f"Proxy-Authorization: Basic {base64.b64encode(secret).decode()}\r\n"
        writer.write((request + "\r\n").encode("latin-1"))
        await writer.drain()

        head = await reader.readuntil(b"\r\n\r\n")
        status = head.split(b"\r\n", 1)[0].split()
        if len(status) < 2 or not status[1].startswith(b"2"):
            _close(writer)
            raise ConnectionError(f"upstream proxy refused: {head.splitlines()[0].decode('latin-1')}")
        return reader, writer

    async def _relay(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        throttled: bool,
    ) -> None:
        """Move bytes one way, paying for them before they are handed on.

        Read first, pay second. Reserving before the read looked tidier and was
        wrong twice over: a connection that ended on the read kept whatever it
        had taken, so every closed tunnel quietly removed up to a chunk from the
        shared budget for good, and a connection sitting idle between range
        requests parked a reservation that the ones with data to move were
        waiting for. Paying for bytes that exist costs nothing that is not
        already in hand: the data waits in memory instead of on the socket, and
        the sender is held back by its own window.
        """
        try:
            while True:
                data = await reader.read(CHUNK)
                if not data:
                    break
                # Read afresh every time: the ceiling can change, or go away,
                # while this tunnel is open.
                bucket = self.bucket if throttled else None
                if bucket is not None:
                    await _pay(bucket, len(data))
                writer.write(data)
                await writer.drain()
        except (OSError, ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            # The peer direction blocks on a read that will never return once
            # this side is done, so close it rather than leave it hanging.
            _close(writer)


async def _pay(bucket: TokenBucket, amount: int) -> None:
    """Block until `amount` bytes have been paid for, in whatever pieces the
    bucket hands out."""
    outstanding = amount
    while outstanding > 0:
        granted, wait = bucket.grant(outstanding)
        if granted:
            outstanding -= granted
        else:
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


def inherited_proxy(env: dict[str, str] | None = None) -> str:
    """The proxy this container was configured with, if any.

    Read from the app's own environment, which the limiter never writes to —
    only the workers' copies are rewritten, and only while a limit is on.
    """
    source = os.environ if env is None else env
    for var in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        value = (source.get(var) or "").strip()
        if value:
            return value
    return ""


class SharedLimit:
    """The one limiter every transfer of this instance goes through.

    Held by the API process: it outlives single jobs, so a changed ceiling
    reaches a download that is already running, and two transfers share one
    budget instead of getting one each.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proxy: ThrottledProxy | None = None
        self._limited = False

    @property
    def url(self) -> str:
        """Where a worker starting now should send its traffic, if anywhere."""
        proxy = self._proxy
        return proxy.url if proxy is not None and self._limited else ""

    @property
    def mbit(self) -> float:
        """The ceiling in force, read from the bucket rather than kept twice."""
        proxy = self._proxy
        if proxy is None or proxy.bucket is None:
            return 0.0
        return proxy.bucket.rate / BYTES_PER_MBIT

    def apply(self, mbit: Any, log: Callable[..., None] | None = None) -> str:
        """Set the ceiling in Mbit/s. Zero — or nonsense — turns it off.

        Returns the proxy URL, or an empty string when nothing is limited. A
        limiter that cannot bind is reported and skipped: a speed setting is not
        worth failing every download over.
        """
        try:
            rate = float(mbit or 0)
        except (TypeError, ValueError):
            rate = 0.0

        with self._lock:
            if rate <= 0:
                # The proxy stays up. Workers that started while the limit was
                # on carry its address for their whole life, and closing the
                # port under them ends their transfer with a refused
                # connection. It lets everything through from here instead, and
                # workers starting from now on are sent out directly.
                if self._proxy is not None:
                    self._proxy.set_rate(None)
                self._limited = False
                return ""

            if self._proxy is not None:
                self._proxy.set_rate(rate * BYTES_PER_MBIT)
                self._limited = True
                return self._proxy.url

            proxy = ThrottledProxy(
                rate * BYTES_PER_MBIT, upstream_proxy=inherited_proxy()
            )
            try:
                url = proxy.start()
            except Exception as exc:  # noqa: BLE001 - never fatal
                if log is not None:
                    log(f"Speed limit of {rate:g} Mbit/s not applied: {exc}")
                return ""
            self._proxy = proxy
            self._limited = True
            return url

    def apply_to_env(self, env: dict[str, str]) -> dict[str, str]:
        """Point a worker's environment at the limiter, if there is one.

        Without a limit the environment is left exactly as it is, so a proxy
        the operator configured for the container stays the worker's proxy.
        """
        url = self.url
        if not url:
            return env
        for var in PROXY_VARS:
            env[var] = url
        for var in NO_PROXY_VARS:
            env.pop(var, None)
        return env

    def stop(self) -> None:
        with self._lock:
            self._shutdown()

    def _shutdown(self) -> None:
        """Caller holds the lock. Only for shutting the app down: while it
        runs, a limit that is switched off keeps its proxy."""
        if self._proxy is not None:
            self._proxy.stop()
            self._proxy = None
        self._limited = False


#: The instance everything else talks to.
shared = SharedLimit()

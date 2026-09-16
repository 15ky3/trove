"""The speed limiter: token maths, the CONNECT proxy, and the env wiring.

Nothing here leaves the machine. The "Hub" is a plain TCP server on localhost
that the proxy tunnels to, which is all a CONNECT proxy ever sees anyway.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import socket
import threading
import time
from urllib.parse import urlsplit

import pytest

from app import throttle


# --------------------------------------------------------------- Token maths


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestTokenBucket:
    def test_hands_out_what_is_there(self):
        bucket = throttle.TokenBucket(1000, burst=500, clock=FakeClock())
        granted, wait = bucket.grant(200)
        assert (granted, wait) == (200, 0.0)

    def test_never_more_than_asked(self):
        bucket = throttle.TokenBucket(1000, burst=5000, clock=FakeClock())
        assert bucket.grant(64)[0] == 64

    def test_empty_bucket_asks_for_a_wait(self):
        clock = FakeClock()
        bucket = throttle.TokenBucket(1000, burst=500, clock=clock)
        bucket.grant(500)
        granted, wait = bucket.grant(500)
        assert granted == 0
        # 500 bytes at 1000 B/s.
        assert wait == pytest.approx(0.5, abs=0.01)

    def test_refills_over_time(self):
        clock = FakeClock()
        bucket = throttle.TokenBucket(1000, burst=500, clock=clock)
        bucket.grant(500)
        clock.advance(0.25)
        assert bucket.grant(500)[0] == 250

    def test_refill_stops_at_the_burst_size(self):
        clock = FakeClock()
        bucket = throttle.TokenBucket(1000, burst=500, clock=clock)
        bucket.grant(500)
        clock.advance(3600)
        # An idle hour does not buy an hour of traffic.
        assert bucket.grant(10_000)[0] == 500

    def test_wait_never_drops_below_the_floor(self):
        clock = FakeClock()
        bucket = throttle.TokenBucket(1_000_000_000, burst=64, clock=clock)
        bucket.grant(64)
        assert bucket.grant(64)[1] >= throttle.MIN_SLEEP

    def test_set_rate_retunes_a_running_bucket(self):
        clock = FakeClock()
        bucket = throttle.TokenBucket(1000, burst=500, clock=clock)
        bucket.grant(500)
        bucket.set_rate(2000)
        clock.advance(0.25)
        assert bucket.grant(10_000)[0] == 500

    def test_set_rate_takes_away_a_burst_that_is_now_too_large(self):
        clock = FakeClock()
        bucket = throttle.TokenBucket(1_000_000, clock=clock)
        # A ceiling lowered mid-transfer must not leave a fat bucket behind.
        bucket.set_rate(1000)
        assert bucket.capacity == throttle.MIN_BURST
        assert bucket.grant(10_000_000)[0] <= throttle.MIN_BURST

    @pytest.mark.parametrize("rate", [0, -1])
    def test_set_rate_refuses_a_rate_that_means_nothing(self, rate):
        bucket = throttle.TokenBucket(1000, clock=FakeClock())
        with pytest.raises(ValueError):
            bucket.set_rate(rate)

    @pytest.mark.parametrize("want", [0, -1])
    def test_asking_for_nothing_grants_nothing(self, want):
        bucket = throttle.TokenBucket(1000, clock=FakeClock())
        assert bucket.grant(want) == (0, 0.0)

    @pytest.mark.parametrize("rate", [0, -1])
    def test_a_rate_that_means_nothing_is_refused(self, rate):
        with pytest.raises(ValueError):
            throttle.TokenBucket(rate)

    def test_default_burst_is_never_smaller_than_a_chunk(self):
        # A slow line must still be able to move one read at a time.
        assert throttle.TokenBucket(10).capacity == throttle.MIN_BURST


class TestPaying:
    def test_pays_the_whole_amount_in_whatever_pieces_it_gets(self):
        bucket = throttle.TokenBucket(100_000, burst=100)
        asyncio.run(throttle._pay(bucket, 300))
        # Everything was paid for: nothing is left to hand out right away.
        assert bucket.grant(300)[0] < 300

    def test_paying_for_nothing_returns_at_once(self):
        bucket = throttle.TokenBucket(1000, burst=500, clock=FakeClock())
        asyncio.run(throttle._pay(bucket, 0))
        assert bucket.grant(500)[0] == 500


# ------------------------------------------------------------ Upstream stand-in


class Upstream:
    """A TCP server on localhost for the proxy to tunnel to."""

    def __init__(self, handler) -> None:
        self.handler = handler
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.host, self.port = self.sock.getsockname()
        self.thread = threading.Thread(target=self._accept, daemon=True)
        self.thread.start()

    def _accept(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        try:
            self.handler(conn)
        except OSError:
            pass
        finally:
            conn.close()

    def close(self) -> None:
        self.sock.close()


def blast(payload: bytes):
    """Sends a fixed body and hangs up."""

    def handler(conn: socket.socket) -> None:
        conn.sendall(payload)

    return handler


def echo(conn: socket.socket) -> None:
    while True:
        data = conn.recv(65536)
        if not data:
            return
        conn.sendall(data)


@pytest.fixture
def upstream():
    servers = []

    def make(handler):
        server = Upstream(handler)
        servers.append(server)
        return server

    yield make
    for server in servers:
        server.close()


@pytest.fixture
def proxy():
    running = []

    def make(rate: float):
        instance = throttle.ThrottledProxy(rate)
        instance.start()
        running.append(instance)
        return instance

    yield make
    for instance in running:
        instance.stop()


def talk(url: str) -> socket.socket:
    parts = urlsplit(url)
    return socket.create_connection((parts.hostname, parts.port), timeout=10)


def read_head(sock: socket.socket) -> str:
    head = b""
    while b"\r\n\r\n" not in head:
        chunk = sock.recv(1)
        if not chunk:
            break
        head += chunk
    return head.decode("latin-1")


def read_all(sock: socket.socket) -> bytes:
    body = b""
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            return body
        body += chunk


def tunnel(proxy_url: str, host: str, port: int) -> tuple[socket.socket, str]:
    sock = talk(proxy_url)
    sock.sendall(f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n".encode())
    return sock, read_head(sock)


# ---------------------------------------------------------------- The tunnel


class TestTunnel:
    def test_relays_a_body_unchanged(self, proxy, upstream):
        payload = os.urandom(200_000)
        server = upstream(blast(payload))
        instance = proxy(50_000_000)

        sock, head = tunnel(instance.url, server.host, server.port)
        assert head.startswith("HTTP/1.1 200 ")
        received = read_all(sock)
        sock.close()

        assert len(received) == len(payload)
        assert hashlib.sha256(received).digest() == hashlib.sha256(payload).digest()

    def test_relays_in_both_directions(self, proxy, upstream):
        server = upstream(echo)
        instance = proxy(50_000_000)

        sock, _ = tunnel(instance.url, server.host, server.port)
        sock.sendall(b"ping")
        assert sock.recv(4) == b"ping"
        sock.close()

    def test_the_limit_is_actually_enforced(self, proxy, upstream):
        # 65_536 bytes come out of the initial burst, the remaining 200_000 are
        # paid for at 200_000 B/s — about a second, whatever the line can do.
        rate = 200_000
        payload = b"x" * (throttle.MIN_BURST + 200_000)
        server = upstream(blast(payload))
        instance = proxy(rate)

        sock, _ = tunnel(instance.url, server.host, server.port)
        started = time.monotonic()
        received = read_all(sock)
        elapsed = time.monotonic() - started
        sock.close()

        assert len(received) == len(payload)
        assert elapsed >= 0.6, f"{len(payload)} bytes arrived in {elapsed:.2f}s — not throttled"

    def test_several_tunnels_share_one_budget(self, proxy, upstream):
        # Xet opens many connections at once; the ceiling is for all of them
        # together, not per connection.
        rate = 200_000
        payload = b"y" * (throttle.MIN_BURST + 100_000)
        server = upstream(blast(payload))
        instance = proxy(rate)

        results: list[int] = []

        def pull() -> None:
            sock, _ = tunnel(instance.url, server.host, server.port)
            results.append(len(read_all(sock)))
            sock.close()

        started = time.monotonic()
        threads = [threading.Thread(target=pull) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        elapsed = time.monotonic() - started

        assert results == [len(payload)] * 3
        # 3 × 165_536 bytes minus one shared burst, at 200_000 B/s.
        assert elapsed >= 1.0, f"three tunnels finished in {elapsed:.2f}s — budget not shared"


class TestBudgetAccounting:
    """What the ceiling is worth over a long transfer, not just a short one."""

    def test_connections_that_end_do_not_burn_the_budget(self, proxy, upstream):
        # Charging before the read meant every tunnel kept what it had taken
        # for the read that returned nothing — up to 64 KiB per connection,
        # gone for good. Xet churns connections, so the rate drifted below the
        # ceiling the longer a transfer ran.
        rate = 50_000
        server = upstream(blast(b"ab"))
        instance = proxy(rate)

        for _ in range(6):
            sock, _ = tunnel(instance.url, server.host, server.port)
            assert read_all(sock) == b"ab"
            sock.close()

        left, _ = instance.bucket.grant(10**9)
        assert left >= instance.bucket.capacity * 0.8, (
            f"only {left} of {instance.bucket.capacity:.0f} tokens left after 12 bytes"
        )

    def test_an_idle_tunnel_does_not_hold_the_budget(self, proxy, upstream):
        # A connection between range requests used to park a reservation the
        # connections with data to move were waiting for.
        rate = 100_000
        idle_server = upstream(lambda conn: time.sleep(5))
        busy_server = upstream(blast(b"z" * throttle.MIN_BURST))
        instance = proxy(rate)

        idle, _ = tunnel(instance.url, idle_server.host, idle_server.port)
        try:
            started = time.monotonic()
            sock, _ = tunnel(instance.url, busy_server.host, busy_server.port)
            received = read_all(sock)
            elapsed = time.monotonic() - started
            sock.close()
        finally:
            idle.close()

        assert len(received) == throttle.MIN_BURST
        # The burst covers it; nothing should have been waiting on the idle one.
        assert elapsed < 0.4, f"a burst-sized transfer waited {elapsed:.2f}s"


class TestRefusals:
    def test_only_connect_is_served(self, proxy):
        instance = proxy(1_000_000)
        sock = talk(instance.url)
        sock.sendall(b"GET http://example.com/ HTTP/1.1\r\nHost: example.com\r\n\r\n")
        assert "405" in read_head(sock)
        sock.close()

    @pytest.mark.parametrize(
        "line",
        [
            b"NONSENSE\r\n\r\n",
            b"CONNECT example.com:nope HTTP/1.1\r\n\r\n",
            b"CONNECT :443 HTTP/1.1\r\n\r\n",
            b"CONNECT example.com:99999 HTTP/1.1\r\n\r\n",
        ],
    )
    def test_a_broken_request_line_is_refused(self, proxy, line):
        instance = proxy(1_000_000)
        sock = talk(instance.url)
        sock.sendall(line)
        assert "400" in read_head(sock)
        sock.close()

    def test_endless_headers_are_cut_off(self, proxy):
        instance = proxy(1_000_000)
        sock = talk(instance.url)
        sock.sendall(b"CONNECT example.com:443 HTTP/1.1\r\n")
        padding = b"X-Pad: " + b"p" * 200 + b"\r\n"
        sock.sendall(padding * (throttle.HEADER_LIMIT // len(padding) + 2))
        assert "400" in read_head(sock)
        sock.close()

    def test_an_upstream_that_is_not_there_reports_502(self, proxy):
        # Bind a port and drop it, so nothing is listening on a port we know.
        spare = socket.socket()
        spare.bind(("127.0.0.1", 0))
        dead_port = spare.getsockname()[1]
        spare.close()

        instance = proxy(1_000_000)
        sock, head = tunnel(instance.url, "127.0.0.1", dead_port)
        assert "502" in head
        sock.close()

    def test_a_client_that_leaves_does_not_take_the_proxy_down(self, proxy, upstream):
        server = upstream(blast(b"z" * 500_000))
        instance = proxy(100_000)

        sock, _ = tunnel(instance.url, server.host, server.port)
        sock.recv(1)
        sock.close()  # hang up mid-transfer

        # Still serving.
        again, head = tunnel(instance.url, server.host, server.port)
        assert head.startswith("HTTP/1.1 200 ")
        again.close()


class TestLifecycle:
    def test_stop_releases_the_port(self, proxy):
        instance = throttle.ThrottledProxy(1_000_000)
        url = instance.start()
        instance.stop()
        with pytest.raises(OSError):
            talk(url).close()

    def test_stop_is_safe_before_a_start(self):
        throttle.ThrottledProxy(1_000_000).stop()


# -------------------------------------------------------- The shared limiter


@pytest.fixture(autouse=True)
def stop_shared():
    yield
    throttle.shared.stop()


class TestSharedLimit:
    def test_off_by_default(self):
        assert throttle.shared.apply(0) == ""
        assert throttle.shared.env() == {}
        assert throttle.shared.mbit == 0.0

    @pytest.mark.parametrize("value", [0, 0.0, None, "", "fast", [], -5])
    def test_nothing_or_nonsense_means_no_limit(self, value):
        assert throttle.shared.apply(value) == ""
        assert throttle.shared.env() == {}

    def test_a_limit_starts_a_proxy_and_names_every_variable(self):
        url = throttle.shared.apply(8)
        assert url.startswith("http://127.0.0.1:")
        assert throttle.shared.env() == {var: url for var in throttle.PROXY_VARS}
        assert throttle.shared.mbit == 8

    def test_mbit_is_a_million_bits(self):
        throttle.shared.apply(8)
        # 8 Mbit/s is a megabyte a second.
        assert throttle.shared._proxy.bucket.rate == pytest.approx(1_000_000)

    def test_a_string_that_is_a_number_still_works(self):
        throttle.shared.apply("2.5")
        assert throttle.shared._proxy.bucket.rate == pytest.approx(2.5 * throttle.BYTES_PER_MBIT)

    def test_changing_the_limit_keeps_the_same_proxy(self):
        # Running transfers point at this port; restarting it would break them.
        first = throttle.shared.apply(10)
        second = throttle.shared.apply(50)
        assert first == second
        assert throttle.shared._proxy.bucket.rate == pytest.approx(50 * throttle.BYTES_PER_MBIT)
        assert throttle.shared.mbit == 50

    def test_setting_it_to_zero_takes_the_proxy_down(self):
        url = throttle.shared.apply(10)
        assert throttle.shared.apply(0) == ""
        assert throttle.shared.env() == {}
        with pytest.raises(OSError):
            talk(url).close()

    def test_switching_it_back_on_binds_again(self):
        throttle.shared.apply(10)
        throttle.shared.apply(0)
        assert throttle.shared.apply(10).startswith("http://127.0.0.1:")

    def test_the_budget_is_shared_by_everything_going_through_it(self, upstream):
        # The point of one limiter for the whole app: two transfers together
        # get the configured rate, not one each.
        throttle.shared.apply(1.6)  # 200_000 B/s
        payload = b"q" * (throttle.MIN_BURST + 100_000)
        server = upstream(blast(payload))
        url = throttle.shared.url

        sizes: list[int] = []

        def pull() -> None:
            sock, _ = tunnel(url, server.host, server.port)
            sizes.append(len(read_all(sock)))
            sock.close()

        started = time.monotonic()
        threads = [threading.Thread(target=pull) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        elapsed = time.monotonic() - started

        assert sizes == [len(payload)] * 2
        assert elapsed >= 0.8, f"two transfers finished in {elapsed:.2f}s — budget not shared"

    def test_a_limiter_that_cannot_bind_is_skipped(self, monkeypatch):
        def refuse(self):
            raise OSError("address already in use")

        monkeypatch.setattr(throttle.ThrottledProxy, "start", refuse)
        lines: list[str] = []

        # No limit is better than no download.
        assert throttle.shared.apply(10, lines.append) == ""
        assert throttle.shared.env() == {}
        assert "address already in use" in lines[0]

    def test_it_works_without_a_log(self):
        assert throttle.shared.apply(1)

    def test_stop_is_idempotent(self):
        throttle.shared.apply(10)
        throttle.shared.stop()
        throttle.shared.stop()
        assert throttle.shared.env() == {}

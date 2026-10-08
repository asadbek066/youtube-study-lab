from __future__ import annotations

import socket

import pytest
import requests

BLOCKED = "real network access"


def test_external_connections_are_blocked() -> None:
    with pytest.raises(RuntimeError, match=BLOCKED):
        socket.create_connection(("93.184.216.34", 80), timeout=1)
    with socket.socket() as sock, pytest.raises(RuntimeError, match=BLOCKED):
        sock.connect(("93.184.216.34", 80))
    with pytest.raises(RuntimeError, match=BLOCKED):
        requests.get("http://203.0.113.5/", timeout=1)


def test_loopback_connections_are_still_allowed() -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=1):
            pass

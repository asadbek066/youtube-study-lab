from __future__ import annotations

import ipaddress
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


class NetworkAccessBlockedError(RuntimeError):
    """Raised when a test tries to reach a non-loopback address."""


def _is_loopback(address: object) -> bool:
    if not isinstance(address, tuple) or not address:
        return True  # AF_UNIX paths and other non-IP addresses stay local.
    host = address[0]
    if isinstance(host, bytes):
        host = host.decode("ascii", "ignore")
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(str(host).split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _block(address: object) -> None:
    raise NetworkAccessBlockedError(
        f"Test attempted real network access to {address!r}; mock it instead."
    )


@pytest.fixture(autouse=True)
def _block_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def guarded_connect(self: socket.socket, address: object) -> None:
        if not _is_loopback(address):
            _block(address)
        return _real_connect(self, address)  # type: ignore[arg-type]

    def guarded_connect_ex(self: socket.socket, address: object) -> int:
        if not _is_loopback(address):
            _block(address)
        return _real_connect_ex(self, address)  # type: ignore[arg-type]

    def guarded_create_connection(address: tuple, *args: object, **kwargs: object):
        if not _is_loopback(address):
            _block(address)
        return _real_create_connection(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)


_real_create_connection = socket.create_connection

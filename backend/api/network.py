"""Working out how other devices on the network should reach this server.

The dashboard cannot reliably build a phone-friendly link from the URL it was
opened with: an operator working on localhost would produce a QR code that
sends the phone back to itself, and a machine with several interfaces can be
reachable at an address the browser never sees. The server knows its own
interfaces, so it answers this question instead of the browser guessing.
"""
from __future__ import annotations

import ipaddress
import socket


def _candidate_addresses() -> list[str]:
    found: list[str] = []

    # Ask the OS which interface it would use to reach the outside world. No
    # packets are sent; this just resolves the routing decision, and it is far
    # more reliable than hostname lookup, which often returns only loopback.
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            found.append(sock.getsockname()[0])
        finally:
            sock.close()
    except OSError:
        pass

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = info[4][0]
            if address not in found:
                found.append(address)
    except OSError:
        pass
    return found


def lan_addresses() -> list[str]:
    """Private IPv4 addresses this machine is reachable at, best first."""
    usable: list[str] = []
    for address in _candidate_addresses():
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            continue
        # Loopback is useless to a phone, and a link-local 169.254.x address
        # means DHCP failed - offering either would just fail silently later.
        if parsed.is_loopback or parsed.is_link_local:
            continue
        if parsed.is_private:
            usable.append(address)
    return usable


def reachable_base_url(scheme: str, port: int) -> str | None:
    """Best URL for another device on this network, or None if there is no
    usable address - in which case the machine is not on a network and no
    amount of QR code will help."""
    addresses = lan_addresses()
    if not addresses:
        return None
    return f"{scheme}://{addresses[0]}:{port}"

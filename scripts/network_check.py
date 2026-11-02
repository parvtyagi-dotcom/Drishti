"""Prove where Drishti's data goes: watch every connection the server makes.

"Does footage leave the post?" deserves evidence, not an assurance. Start
Drishti, use it as normal (or run the demo), and in a second terminal:

    python scripts/network_check.py              # watch for 60 seconds
    python scripts/network_check.py --seconds 300

It finds the running Drishti server, samples its network connections once a
second, and reports every remote address it talked to, sorted into this
machine, the local network, and the internet. With the AI analyst on and a
cloud model configured, the internet line names those calls, which carry
event text only, never video. With cloud AI off, that line should be empty.
"""
from __future__ import annotations

import argparse
import ipaddress
import socket
import sys
import time
from collections import defaultdict

import psutil

MARKERS = ("scripts/serve.py", "scripts\\serve.py", "backend.app", "uvicorn")


def drishti_processes() -> list[psutil.Process]:
    found = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        cmdline = " ".join(proc.info.get("cmdline") or [])
        if "python" in (proc.info.get("name") or "").lower() and any(m in cmdline for m in MARKERS) \
                and "network_check" not in cmdline:
            found.append(proc)
    return found


def classify(address: str) -> str:
    ip = ipaddress.ip_address(address)
    if ip.is_loopback:
        return "this machine"
    if ip.is_private or ip.is_link_local or ip.is_multicast:
        return "local network"
    return "internet"


def name_for(address: str) -> str:
    try:
        return socket.gethostbyaddr(address)[0]
    except OSError:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--pid", type=int, help="watch this process instead of finding the server")
    args = parser.parse_args()

    procs = [psutil.Process(args.pid)] if args.pid else drishti_processes()
    # A launcher or shell wrapper may hold the PID while the server runs as
    # its child, so the whole process tree is watched.
    procs = list({p.pid: p for root in procs for p in [root, *root.children(recursive=True)]}.values())
    if not procs:
        print("No running Drishti server found. Start it with: python scripts/serve.py")
        return 1
    print(f"Watching {len(procs)} Drishti process(es) (pid {', '.join(str(p.pid) for p in procs)}) "
          f"for {args.seconds} seconds. Use the dashboard as normal meanwhile.\n")

    seen: dict[tuple[str, int], set[str]] = defaultdict(set)
    listening: set[tuple[str, int]] = set()
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        for proc in procs:
            try:
                connections = proc.net_connections(kind="inet")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            for c in connections:
                if c.status == psutil.CONN_LISTEN and c.laddr:
                    listening.add((c.laddr.ip, c.laddr.port))
                elif c.raddr:
                    seen[(c.raddr.ip, c.raddr.port)].add(c.status)
        time.sleep(1)

    print("Listening for:", ", ".join(f"{ip}:{port}" for ip, port in sorted(listening)) or "nothing")
    groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (ip, port) in seen:
        groups[classify(ip)].append((ip, port))
    for group in ("this machine", "local network", "internet"):
        items = sorted(groups.get(group, []))
        print(f"\n{group.capitalize()}: {len(items)} address(es)")
        for ip, port in items[:20]:
            label = name_for(ip) if group == "internet" else ""
            print(f"  {ip}:{port}" + (f"  ({label})" if label else ""))
    outside = groups.get("internet", [])
    print("\nVerdict:", "nothing left this network during the check." if not outside else
          f"{len(outside)} internet connection(s). If the AI analyst uses a cloud model these are expected "
          "(event text only); otherwise investigate them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

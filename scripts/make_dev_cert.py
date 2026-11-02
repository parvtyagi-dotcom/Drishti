"""Generate a self-signed TLS certificate for LAN access.

Why this exists: a browser only exposes a device camera on a *secure* origin.
`http://localhost` counts as secure, but `http://192.168.x.x` - which is what a
phone must use to reach this machine - does not. So the phone-camera page is
unreachable over plain HTTP on a LAN, and the server needs TLS.

The certificate is self-signed, so the phone shows a warning the first time and
the operator accepts it once. That is fine for a local deployment on a closed
network and is not a substitute for a real certificate in production.

Usage:
    python scripts/make_dev_cert.py
    uvicorn backend.app:app --host 0.0.0.0 --port 8443 \
        --ssl-keyfile certs/dev-key.pem --ssl-certfile certs/dev-cert.pem

Then open https://<this-machine-ip>:8443/api/live/siren on the phone.

Note that binding 0.0.0.0 exposes the API to everything on the network. The API
has no authentication yet, so only do this on a network you control.
"""
from __future__ import annotations

import ipaddress
import shutil
import socket
import subprocess
import sys
from pathlib import Path

CERT_DIR = Path(__file__).resolve().parent.parent / "certs"
KEY_PATH = CERT_DIR / "dev-key.pem"
CERT_PATH = CERT_DIR / "dev-cert.pem"
DAYS = 825  # longest span browsers still accept for a leaf certificate


def local_ip_addresses() -> list[str]:
    """Best-effort list of this machine's LAN addresses.

    The UDP-connect trick asks the OS which interface it would use to reach the
    outside world, without sending anything - far more reliable than
    gethostbyname, which often returns only 127.0.0.1.
    """
    found: list[str] = []
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
            addr = info[4][0]
            if addr not in found:
                found.append(addr)
    except OSError:
        pass

    usable = []
    for addr in found:
        try:
            parsed = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if not parsed.is_loopback:
            usable.append(addr)
    return usable


def main() -> int:
    openssl = shutil.which("openssl")
    if not openssl:
        print("openssl not found on PATH. Git for Windows ships it - try running this from Git Bash.")
        return 1

    ips = local_ip_addresses()
    CERT_DIR.mkdir(parents=True, exist_ok=True)

    # Modern browsers ignore the certificate's Common Name entirely and match
    # only against subjectAltName, so every address the phone might dial has to
    # be listed here or the certificate is rejected outright.
    alt_names = ["DNS:localhost", "IP:127.0.0.1"] + [f"IP:{ip}" for ip in ips]
    subject = "/C=IN/O=Drishti/CN=border-watch.local"

    cmd = [
        openssl, "req", "-x509", "-newkey", "rsa:2048", "-sha256",
        "-days", str(DAYS), "-nodes",
        "-keyout", str(KEY_PATH), "-out", str(CERT_PATH),
        "-subj", subject,
        "-addext", f"subjectAltName={','.join(alt_names)}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("openssl failed:\n" + (result.stderr or result.stdout))
        return result.returncode

    print(f"certificate: {CERT_PATH}")
    print(f"private key: {KEY_PATH}")
    print("valid for:   " + ", ".join(alt_names))
    print()
    print("Start the server with TLS:")
    print("  uvicorn backend.app:app --host 0.0.0.0 --port 8443 \\")
    print(f"      --ssl-keyfile {KEY_PATH.relative_to(CERT_DIR.parent)} \\")
    print(f"      --ssl-certfile {CERT_PATH.relative_to(CERT_DIR.parent)}")
    print()
    if ips:
        print("Then on the phone (accept the certificate warning once):")
        for ip in ips:
            print(f"  siren:  https://{ip}:8443/api/live/siren")
            print(f"  camera: https://{ip}:8443/api/live/phone?camera_id=<id>")
    else:
        print("No LAN address detected - is this machine connected to a network?")
    return 0


if __name__ == "__main__":
    sys.exit(main())

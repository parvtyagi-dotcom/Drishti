"""Start Drishti for real use: one command, both protocols.

Two listeners are needed at once, for reasons that are not arbitrary:

  http  (8000) - the Android siren app. A native app cannot show the
                 "accept this certificate" prompt a browser offers, so a
                 self-signed certificate simply fails there. Plain HTTP on the
                 local network is what actually works.
  https (8443) - browsers. A phone will not grant camera access to a page
                 served over plain HTTP at a LAN address, so pairing a phone
                 as a camera requires TLS.

Running one or the other means something always appears broken, so this starts
both against the same application state.

    python scripts/serve.py            # both, on all interfaces
    python scripts/serve.py --http-only

Binding to all interfaces exposes the API to the local network. Authentication
is on by default; keep it that way on any network you do not control.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Google sign-in for the demo ("Drishti Login" project). A client ID is
# public - it ships in the frontend bundle - so defaulting it here is safe.
# Set GOOGLE_CLIENT_ID= (empty) to start without sign-in.
os.environ.setdefault(
    "GOOGLE_CLIENT_ID", "989388168434-828tilr4ac5pp3qirvjou30iag4didca.apps.googleusercontent.com"
)

import uvicorn  # noqa: E402

from backend.api.network import lan_addresses  # noqa: E402

CERT = REPO / "certs" / "dev-cert.pem"
KEY = REPO / "certs" / "dev-key.pem"


def serve(port: int, tls: bool, host: str) -> None:
    config = uvicorn.Config(
        "backend.app:app", host=host, port=port, log_level="info",
        ssl_certfile=str(CERT) if tls else None,
        ssl_keyfile=str(KEY) if tls else None,
    )
    uvicorn.Server(config).run()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--http-port", type=int, default=8000)
    parser.add_argument("--https-port", type=int, default=8443)
    parser.add_argument("--http-only", action="store_true")
    args = parser.parse_args()

    tls_available = CERT.is_file() and KEY.is_file()
    if not tls_available and not args.http_only:
        print("No certificate found. Run: python scripts/make_dev_cert.py")
        print("Continuing with HTTP only - phone cameras will not work.\n")

    addresses = lan_addresses()
    primary = addresses[0] if addresses else socket.gethostbyname(socket.gethostname())

    print("=" * 62)
    print("  BORDER-WATCH")
    print("=" * 62)
    print(f"  Dashboard (browser)   https://{primary}:{args.https_port}"
          if tls_available and not args.http_only else
          f"  Dashboard (browser)   http://{primary}:{args.http_port}")
    print(f"  Siren app (Android)   {primary}:{args.http_port}")
    print(f"  Siren page (browser)  http://{primary}:{args.http_port}/api/live/siren")
    if not addresses:
        print("\n  WARNING: no network address found - other devices cannot reach this machine.")
    print("=" * 62 + "\n")

    if tls_available and not args.http_only:
        # Tell the app that a TLS listener exists and on which port. A request
        # arriving over http cannot discover this for itself, and without it
        # the dashboard tells the operator to enable TLS that is already
        # running - which is exactly the wrong instruction.
        os.environ["BORDERWATCH_HTTPS_PORT"] = str(args.https_port)
        # HTTPS on a background thread; HTTP stays on the main thread so
        # Ctrl+C reaches it and stops everything.
        threading.Thread(
            target=serve, args=(args.https_port, True, args.host), daemon=True,
        ).start()
    serve(args.http_port, False, args.host)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nstopped")

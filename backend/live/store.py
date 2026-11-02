"""Durable, tamper-evident event store.

Two problems solved together, because they are the same problem.

*Persistence*: events lived only in memory, so a restart erased the record
while the evidence clips it referred to stayed on disk. A security log that
forgets is not a log.

*Tamper-evidence*: a stored log that anyone can quietly edit proves nothing
afterwards. Each row therefore carries the hash of the row before it, so the
whole history forms a chain: changing, inserting or deleting any record breaks
every hash after it, and the break is detectable without a trusted third
party.

This is deliberately not a blockchain. There is no network, no consensus and
no token - those solve mutual distrust between organisations, which is not the
situation here. The property actually needed is "nobody can alter the record
without it being obvious", and a hash chain gives exactly that. A permissioned
ledger can be layered on later if a deployment ever needs several parties to
distrust each other's copies.

Writes happen on a background thread. An analysis thread must never wait on a
disk flush to report that someone crossed a fence.
"""
from __future__ import annotations

import hashlib
import json
import logging
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("backend.live.store")

GENESIS_HASH = "0" * 64

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id      TEXT    NOT NULL UNIQUE,
    camera_id     TEXT    NOT NULL,
    camera_name   TEXT,
    type          TEXT    NOT NULL,
    severity      TEXT    NOT NULL,
    reason        TEXT,
    alarm         INTEGER NOT NULL DEFAULT 0,
    track_id      INTEGER,
    zone          TEXT,
    wall_time     REAL    NOT NULL,
    timestamp_sec REAL,
    evidence      TEXT,
    payload       TEXT    NOT NULL,
    prev_hash     TEXT    NOT NULL,
    hash          TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_wall_time ON events(wall_time);
CREATE INDEX IF NOT EXISTS idx_events_camera    ON events(camera_id);
CREATE INDEX IF NOT EXISTS idx_events_type      ON events(type);
CREATE INDEX IF NOT EXISTS idx_events_alarm     ON events(alarm);
"""


def _canonical(event: dict) -> str:
    """Stable serialisation of the fields the hash covers.

    Sorted keys and no whitespace, so the same event always hashes the same
    way regardless of dict ordering - otherwise verification would fail on
    records nobody touched.
    """
    covered = {
        "event_id": event.get("event_id"),
        "camera_id": event.get("camera_id"),
        "type": event.get("type"),
        "severity": event.get("severity"),
        "reason": event.get("reason"),
        "alarm": bool(event.get("alarm")),
        "track_id": event.get("track_id"),
        "zone": event.get("zone"),
        "wall_time": event.get("wall_time"),
        "timestamp_sec": event.get("timestamp_sec"),
        "evidence_bundle": event.get("evidence_bundle"),
        "data": event.get("data"),
    }
    return json.dumps(covered, sort_keys=True, separators=(",", ":"), default=str)


def chain_hash(prev_hash: str, event: dict) -> str:
    return hashlib.sha256((prev_hash + _canonical(event)).encode("utf-8")).hexdigest()


@dataclass
class ChainStatus:
    total_events: int
    valid: bool
    broken_at_id: int | None = None
    broken_event_id: str | None = None
    detail: str | None = None

    def as_dict(self) -> dict:
        return {
            "total_events": self.total_events,
            "valid": self.valid,
            "broken_at_id": self.broken_at_id,
            "broken_event_id": self.broken_event_id,
            "detail": self.detail,
        }


class EventStore:
    """SQLite-backed event log with a hash chain over its rows."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Anchor kept OUTSIDE the database. A hash chain proves that no record
        # was modified or removed from the middle, because every later hash
        # would stop matching - but deleting the most recent records leaves a
        # shorter chain that is still internally consistent. Recording the
        # expected length and head hash separately is what makes that
        # truncation visible: an attacker must now alter two artifacts that
        # live in different places, and doing only one is detected.
        self.anchor_path = self.path.with_name("chain-head.json")
        self._queue: queue.Queue[dict | None] = queue.Queue(maxsize=5000)
        self._writer: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_hash = GENESIS_HASH
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        # WAL lets the dashboard read while the writer thread appends, instead
        # of the two blocking each other.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            row = conn.execute("SELECT hash FROM events ORDER BY id DESC LIMIT 1").fetchone()
            # Resume the existing chain across restarts; starting a new one
            # would leave an unverifiable seam in the history.
            self._last_hash = row["hash"] if row else GENESIS_HASH

    # --- writing -------------------------------------------------------------

    def start(self) -> None:
        if self._writer is not None and self._writer.is_alive():
            return
        self._stop.clear()
        self._writer = threading.Thread(target=self._writer_loop, name="event-store", daemon=True)
        self._writer.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._writer is not None:
            self._writer.join(timeout=5.0)
            self._writer = None

    def record(self, event: dict) -> None:
        """Queue an event for durable storage. Never blocks the caller."""
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            # Dropping is better than stalling analysis, but it must be loud:
            # a silently truncated security log is worse than a noisy one.
            logger.error("event store queue full - dropping event %s", event.get("event_id"))

    def _writer_loop(self) -> None:
        conn = self._connect()
        try:
            while not self._stop.is_set():
                try:
                    event = self._queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                if event is None:
                    break
                try:
                    self._insert(conn, event)
                except Exception:
                    logger.exception("failed to persist event %s", event.get("event_id"))
                finally:
                    self._queue.task_done()
        finally:
            conn.close()

    def _insert(self, conn: sqlite3.Connection, event: dict) -> None:
        with self._lock:
            prev = self._last_hash
            digest = chain_hash(prev, event)
            cursor = conn.execute(
                """INSERT OR IGNORE INTO events
                   (event_id, camera_id, camera_name, type, severity, reason, alarm,
                    track_id, zone, wall_time, timestamp_sec, evidence, payload, prev_hash, hash)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    event.get("event_id"), event.get("camera_id"), event.get("camera_name"),
                    event.get("type"), event.get("severity"), event.get("reason"),
                    1 if event.get("alarm") else 0, event.get("track_id"), event.get("zone"),
                    event.get("wall_time", time.time()), event.get("timestamp_sec"),
                    event.get("evidence_bundle"),
                    json.dumps(event, default=str), prev, digest,
                ),
            )
            if cursor.rowcount == 0:
                # A duplicate event_id was ignored, so nothing was stored.
                # Advancing the head here would chain the next record to a
                # hash that exists nowhere in the table, breaking verification
                # permanently for every record after it.
                conn.rollback()
                logger.warning("duplicate event_id ignored: %s", event.get("event_id"))
                return
            conn.commit()
            self._last_hash = digest
            self._write_anchor(conn)

    def _write_anchor(self, conn: sqlite3.Connection) -> None:
        try:
            row = conn.execute("SELECT COUNT(*) AS n, MAX(id) AS last_id FROM events").fetchone()
            self.anchor_path.write_text(json.dumps({
                "count": row["n"], "last_id": row["last_id"],
                "head_hash": self._last_hash, "updated_at": time.time(),
            }), encoding="utf-8")
        except (OSError, sqlite3.Error):
            logger.exception("could not update chain anchor")

    def read_anchor(self) -> dict | None:
        try:
            return json.loads(self.anchor_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    # --- reading -------------------------------------------------------------

    def query(self, limit: int = 100, camera_id: str | None = None,
              severity: str | None = None, alarms_only: bool = False,
              since: float | None = None, types: list[str] | None = None) -> list[dict]:
        sql = "SELECT * FROM events WHERE 1=1"
        args: list = []
        if camera_id:
            sql += " AND camera_id = ?"
            args.append(camera_id)
        if types:
            # Indexed column: a week of plate reads is found without scanning
            # every routine detection in between.
            sql += f" AND type IN ({','.join('?' * len(types))})"
            args.extend(types)
        if severity:
            sql += " AND severity = ?"
            args.append(severity)
        if alarms_only:
            sql += " AND alarm = 1"
        if since is not None:
            sql += " AND wall_time >= ?"
            args.append(since)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(max(1, min(limit, 1000)))

        with self._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        out = []
        for row in rows:
            try:
                payload = json.loads(row["payload"])
            except ValueError:
                payload = {}
            out.append({**payload, "row_id": row["id"], "hash": row["hash"], "prev_hash": row["prev_hash"]})
        return out

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"]

    # --- integrity -----------------------------------------------------------

    def verify_chain(self) -> ChainStatus:
        """Recompute every hash and report the first record that does not match.

        This is the whole point of the chain: an altered, inserted or removed
        row cannot keep the sequence consistent, so tampering is detectable
        rather than merely discouraged.
        """
        prev = GENESIS_HASH
        checked = 0
        with self._connect() as conn:
            for row in conn.execute("SELECT * FROM events ORDER BY id ASC"):
                checked += 1
                try:
                    payload = json.loads(row["payload"])
                except ValueError:
                    return ChainStatus(checked, False, row["id"], row["event_id"],
                                       "stored payload is not readable JSON")
                if row["prev_hash"] != prev:
                    return ChainStatus(checked, False, row["id"], row["event_id"],
                                       "record does not follow the previous one "
                                       "(a record was removed, reordered or inserted)")
                expected = chain_hash(prev, payload)
                if expected != row["hash"]:
                    return ChainStatus(checked, False, row["id"], row["event_id"],
                                       "record contents do not match its hash "
                                       "(this record was modified after it was written)")
                prev = row["hash"]

        # The chain itself is consistent. Now check it against the anchor: a
        # shorter-but-valid chain means the newest records were removed, which
        # the chain alone cannot reveal.
        anchor = self.read_anchor()
        if anchor is not None:
            expected = anchor.get("count", 0)
            if checked < expected:
                return ChainStatus(
                    checked, False, None, None,
                    f"{expected - checked} of the most recent records are missing "
                    f"(expected {expected}, found {checked}) - the log was truncated",
                )
            if checked == expected and anchor.get("head_hash") not in (prev, GENESIS_HASH):
                return ChainStatus(
                    checked, False, None, None,
                    "the final record does not match the recorded head - "
                    "the end of the log was replaced",
                )
        return ChainStatus(checked, True)

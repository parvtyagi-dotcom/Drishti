"""Durability and tamper-evidence of the event store.

The tampering tests deliberately edit the database directly, because that is
the threat the hash chain exists to address - someone with access to the file
quietly rewriting history. A test that only used the store's own API would
never exercise it.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from backend.live.store import GENESIS_HASH, EventStore, chain_hash


def make_event(n: int, **overrides) -> dict:
    event = {
        "event_id": f"evt{n:04d}",
        "camera_id": "cam-01",
        "camera_name": "CAM-01",
        "type": "ZONE_ENTRY",
        "severity": "MEDIUM",
        "reason": f"Track #{n} entered 'BORDER LINE'",
        "alarm": False,
        "track_id": n,
        "zone": "zone-1",
        "wall_time": 1_700_000_000.0 + n,
        "timestamp_sec": float(n),
        "data": {"n": n},
    }
    event.update(overrides)
    return event


@pytest.fixture
def store(tmp_path):
    store = EventStore(tmp_path / "events.db")
    yield store
    store.stop()


def write_all(store: EventStore, events: list[dict]) -> None:
    """Write synchronously so tests never race the background writer."""
    conn = store._connect()
    try:
        for event in events:
            store._insert(conn, event)
    finally:
        conn.close()


def test_events_survive_a_restart(tmp_path):
    path = tmp_path / "events.db"
    first = EventStore(path)
    write_all(first, [make_event(i) for i in range(5)])
    first.stop()

    # A new process opening the same file must see the same history - the
    # whole point of persisting rather than keeping events in memory.
    second = EventStore(path)
    assert second.count() == 5
    assert second.verify_chain().valid
    second.stop()


def test_chain_resumes_rather_than_restarting(tmp_path):
    path = tmp_path / "events.db"
    first = EventStore(path)
    write_all(first, [make_event(i) for i in range(3)])
    head = first._last_hash
    first.stop()

    second = EventStore(path)
    assert second._last_hash == head  # continues the chain across restarts
    write_all(second, [make_event(i) for i in range(3, 6)])
    assert second.verify_chain().valid
    second.stop()


def test_intact_log_verifies(store):
    write_all(store, [make_event(i) for i in range(20)])
    status = store.verify_chain()
    assert status.valid
    assert status.total_events == 20


def test_modified_record_is_detected(store):
    write_all(store, [make_event(i) for i in range(10)])
    conn = sqlite3.connect(store.path)
    row = conn.execute("SELECT id, payload FROM events ORDER BY id LIMIT 1 OFFSET 4").fetchone()
    payload = json.loads(row[1])
    payload["reason"] = "Nothing happened here"
    conn.execute("UPDATE events SET payload=?, reason=? WHERE id=?",
                 (json.dumps(payload), "Nothing happened here", row[0]))
    conn.commit()
    conn.close()

    status = store.verify_chain()
    assert not status.valid
    assert status.broken_at_id == row[0]
    assert "modified" in (status.detail or "")


def test_record_removed_from_the_middle_is_detected(store):
    write_all(store, [make_event(i) for i in range(10)])
    conn = sqlite3.connect(store.path)
    victim = conn.execute("SELECT id FROM events ORDER BY id LIMIT 1 OFFSET 5").fetchone()[0]
    conn.execute("DELETE FROM events WHERE id=?", (victim,))
    conn.commit()
    conn.close()

    status = store.verify_chain()
    assert not status.valid


def test_truncating_the_newest_records_is_detected(store):
    """A shortened chain is still internally consistent, so the chain alone
    cannot catch this - the separate head anchor is what does."""
    write_all(store, [make_event(i) for i in range(30)])
    assert store.verify_chain().valid

    conn = sqlite3.connect(store.path)
    conn.execute("DELETE FROM events WHERE id > 10")
    conn.commit()
    conn.close()

    status = store.verify_chain()
    assert not status.valid
    assert "truncated" in (status.detail or "")


def test_hash_depends_on_content_and_predecessor():
    event = make_event(1)
    assert chain_hash(GENESIS_HASH, event) == chain_hash(GENESIS_HASH, event)
    assert chain_hash(GENESIS_HASH, event) != chain_hash("f" * 64, event)
    assert chain_hash(GENESIS_HASH, event) != chain_hash(GENESIS_HASH, make_event(1, reason="other"))


def test_hash_is_stable_regardless_of_key_order():
    # Verification recomputes hashes from JSON that may deserialise in a
    # different order; unstable hashing would flag untouched records.
    a = make_event(7)
    b = {k: a[k] for k in reversed(list(a.keys()))}
    assert chain_hash(GENESIS_HASH, a) == chain_hash(GENESIS_HASH, b)


def test_query_filters(store):
    write_all(store, [
        make_event(1, severity="HIGH", alarm=True, type="TARGET_CONFIRMED"),
        make_event(2, severity="LOW"),
        make_event(3, camera_id="cam-02"),
    ])
    assert len(store.query(limit=50)) == 3
    assert len(store.query(camera_id="cam-02")) == 1
    assert len(store.query(severity="HIGH")) == 1
    assert len(store.query(alarms_only=True)) == 1


def test_duplicate_event_ids_do_not_corrupt_the_chain(store):
    write_all(store, [make_event(1), make_event(1), make_event(2)])
    # The duplicate is ignored rather than stored twice, and what remains
    # still verifies.
    assert store.verify_chain().valid

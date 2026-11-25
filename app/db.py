"""SQLite connection, schema and transaction management."""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import DB_PATH

_write_lock = threading.RLock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS meters (
    meter_id    TEXT PRIMARY KEY,
    parent_id   TEXT REFERENCES meters(meter_id),
    name        TEXT,
    sort_order  INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_meters_parent ON meters(parent_id, sort_order, meter_id);

CREATE TABLE IF NOT EXISTS readings (
    meter_id    TEXT NOT NULL REFERENCES meters(meter_id),
    hour        TEXT NOT NULL,
    amount      INTEGER NOT NULL CHECK (amount >= 0),
    version     INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    PRIMARY KEY (meter_id, hour)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_readings_hour ON readings(hour, meter_id);

CREATE TABLE IF NOT EXISTS reading_events (
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    meter_id    TEXT NOT NULL REFERENCES meters(meter_id),
    hour        TEXT NOT NULL,
    action      TEXT NOT NULL CHECK (action IN ('report', 'correct', 'withdraw')),
    amount      INTEGER CHECK (amount IS NULL OR amount >= 0),
    version     INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_reading_events_lookup
    ON reading_events(hour, meter_id, event_id);

CREATE TABLE IF NOT EXISTS reconciliations (
    hour                TEXT NOT NULL,
    meter_id            TEXT NOT NULL REFERENCES meters(meter_id),
    own_amount          INTEGER NOT NULL CHECK (own_amount >= 0),
    is_complete         INTEGER NOT NULL CHECK (is_complete IN (0, 1)),
    difference          INTEGER,
    sum_used            INTEGER NOT NULL,
    used_meter_ids      TEXT NOT NULL,
    used_readings_json  TEXT NOT NULL,
    missing_meter_ids   TEXT NOT NULL,
    computed_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    version             INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (hour, meter_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_reconciliations_hour_order
    ON reconciliations(hour, meter_id);

-- Fast repeated-correction bookkeeping. These tables are additive so an old
-- database file can be opened and corrected directly: init_db creates them.
CREATE TABLE IF NOT EXISTS app_meta (
    key          TEXT PRIMARY KEY,
    value        TEXT NOT NULL
);
INSERT OR IGNORE INTO app_meta(key, value) VALUES ('tree_version', '0');

-- coverage_version counts membership changes (report inserted / reading
-- withdrawn) for one hour. Pure amount corrections do not bump it.
CREATE TABLE IF NOT EXISTS hour_coverage (
    hour             TEXT PRIMARY KEY,
    coverage_version INTEGER NOT NULL DEFAULT 0
) WITHOUT ROWID;

-- Records the tree/coverage versions under which a meter's last full chain
-- recomputation was performed, plus the chain and which nodes had conclusions.
CREATE TABLE IF NOT EXISTS correction_basis (
    meter_id             TEXT NOT NULL,
    hour                 TEXT NOT NULL,
    tree_version         INTEGER NOT NULL,
    coverage_version     INTEGER NOT NULL,
    chain_ids_json       TEXT NOT NULL,
    conclusion_ids_json  TEXT NOT NULL,
    created_at           TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at           TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    PRIMARY KEY (meter_id, hour)
) WITHOUT ROWID;

-- Triggers keep the change tokens honest for every write path, including
-- direct SQL/reset scripts, so a stale fast path can never be used after the
-- tree or the hour's reported-meter set changed.
CREATE TRIGGER IF NOT EXISTS trg_meters_tree_insert
AFTER INSERT ON meters
BEGIN
    UPDATE app_meta
       SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)
     WHERE key = 'tree_version';
END;

CREATE TRIGGER IF NOT EXISTS trg_meters_tree_delete
AFTER DELETE ON meters
BEGIN
    UPDATE app_meta
       SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)
     WHERE key = 'tree_version';
END;

CREATE TRIGGER IF NOT EXISTS trg_meters_tree_update
AFTER UPDATE ON meters
WHEN OLD.parent_id IS NOT NEW.parent_id
  OR OLD.sort_order IS NOT NEW.sort_order
BEGIN
    UPDATE app_meta
       SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)
     WHERE key = 'tree_version';
END;

CREATE TRIGGER IF NOT EXISTS trg_readings_cov_insert
AFTER INSERT ON readings
BEGIN
    INSERT INTO hour_coverage(hour, coverage_version)
    VALUES (NEW.hour, 1)
    ON CONFLICT(hour) DO UPDATE SET
        coverage_version = hour_coverage.coverage_version + 1;
END;

CREATE TRIGGER IF NOT EXISTS trg_readings_cov_delete
AFTER DELETE ON readings
BEGIN
    INSERT INTO hour_coverage(hour, coverage_version)
    VALUES (OLD.hour, 1)
    ON CONFLICT(hour) DO UPDATE SET
        coverage_version = hour_coverage.coverage_version + 1;
END;

-- Only primary-key changes can alter the reported-meter set; amount/version
-- updates (the correction hot path) deliberately do not fire this.
CREATE TRIGGER IF NOT EXISTS trg_readings_cov_key_update
AFTER UPDATE OF meter_id, hour ON readings
BEGIN
    INSERT INTO hour_coverage(hour, coverage_version)
    VALUES (OLD.hour, 1)
    ON CONFLICT(hour) DO UPDATE SET
        coverage_version = hour_coverage.coverage_version + 1;
    INSERT INTO hour_coverage(hour, coverage_version)
    VALUES (NEW.hour, 1)
    ON CONFLICT(hour) DO UPDATE SET
        coverage_version = hour_coverage.coverage_version + 1;
END;
"""


def connect() -> sqlite3.Connection:
    if isinstance(DB_PATH, Path):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    elif DB_PATH != ":memory:":
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(DB_PATH),
        timeout=30.0,
        isolation_level=None,  # explicit BEGIN IMMEDIATE
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA synchronous = FULL")
    return conn


def init_db() -> None:
    with _write_lock:
        conn = connect()
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(SCHEMA)
        finally:
            conn.close()


@contextmanager
def read_connection() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def write_transaction() -> Iterator[sqlite3.Connection]:
    """Serialize writers and make ledger + conclusion commit atomic."""
    with _write_lock:
        conn = connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            # Allow adjacency-list payloads in any order; all FKs are checked
            # at commit, after every meter row has been inserted.
            conn.execute("PRAGMA defer_foreign_keys = ON")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()


def reset_database_file() -> None:
    """Test/reset helper. Removes all rows while preserving schema."""
    # Additively upgrade old database files first so the bookkeeping tables
    # touched below always exist.
    init_db()
    with write_transaction() as conn:
        conn.execute("DELETE FROM reconciliations")
        conn.execute("DELETE FROM reading_events")
        conn.execute("DELETE FROM readings")
        conn.execute("DELETE FROM meters")
        # Fast-path bookkeeping is derived state; reset it as well. The meter
        # and reading deletes fire version triggers, so reset tokens afterwards.
        conn.execute("DELETE FROM correction_basis")
        conn.execute("DELETE FROM hour_coverage")
        conn.execute("UPDATE app_meta SET value = '0' WHERE key = 'tree_version'")

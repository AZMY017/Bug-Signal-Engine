"""The engine's OWN small Neon database (RESULTS_DATABASE_URL): scan runs, found bugs, settings, alerts."""
import os
import time

import psycopg

URL = os.environ.get("RESULTS_DATABASE_URL", "").strip()

DDL = [
    "CREATE TABLE IF NOT EXISTS runs (id serial PRIMARY KEY, started bigint, finished bigint, status text, params text, note text)",
    """CREATE TABLE IF NOT EXISTS bugs (
        run_id int NOT NULL, pair text NOT NULL, minute smallint NOT NULL, side text NOT NULL, mtg smallint NOT NULL,
        hs smallint NOT NULL, he smallint NOT NULL, nloss smallint NOT NULL,
        n int, w int, ntr int, wtr int, nte int, wte int, maxl smallint, maxl_tr smallint, maxl_te smallint,
        base real, score real, robust boolean,
        PRIMARY KEY (run_id, pair, minute, side, mtg, hs, he, nloss))""",
    "CREATE INDEX IF NOT EXISTS bugs_rank ON bugs (run_id, robust, score DESC)",
    "CREATE TABLE IF NOT EXISTS kv (k text PRIMARY KEY, v text NOT NULL)",
    """CREATE TABLE IF NOT EXISTS alerts (
        key text PRIMARY KEY, created bigint, slot_ts bigint, pair text, side text, mtg smallint,
        state text, step smallint, rule text)""",
]
COLS = ("pair", "minute", "side", "mtg", "hs", "he", "nloss", "n", "w", "ntr", "wtr", "nte", "wte",
        "maxl", "maxl_tr", "maxl_te", "base", "score", "robust")


def enabled():
    return bool(URL)


def _c():
    return psycopg.connect(URL, connect_timeout=30)


def init():
    with _c() as c:
        for q in DDL:
            c.execute(q)


# ---- scan runs
def new_run(params):
    with _c() as c:
        return c.execute("INSERT INTO runs (started, status, params) VALUES (%s,'running',%s) RETURNING id",
                         (int(time.time()), params)).fetchone()[0]


def add_rows(run_id, rows):
    if not rows:
        return
    q = "INSERT INTO bugs (run_id, %s) VALUES (%%s, %s) ON CONFLICT DO NOTHING" % (
        ",".join(COLS), ",".join(["%s"] * len(COLS)))
    data = [(run_id,) + tuple(r[k] for k in COLS) for r in rows]
    with _c() as c:
        with c.cursor() as cur:
            cur.executemany(q, data)


def finish_run(run_id, status, note):
    with _c() as c:
        c.execute("UPDATE runs SET finished=%s, status=%s, note=%s WHERE id=%s", (int(time.time()), status, note, run_id))
        # keep the 3 newest finished runs only
        c.execute("DELETE FROM bugs WHERE run_id NOT IN (SELECT id FROM runs WHERE status='done' ORDER BY id DESC LIMIT 3) "
                  "AND run_id <> %s", (run_id,))


def latest_run():
    """(id, started, finished, note) of the newest finished scan, or None."""
    with _c() as c:
        return c.execute("SELECT id, started, finished, note FROM runs WHERE status='done' ORDER BY id DESC LIMIT 1").fetchone()


def previous_run(before_id):
    with _c() as c:
        row = c.execute("SELECT id FROM runs WHERE status='done' AND id<%s ORDER BY id DESC LIMIT 1", (before_id,)).fetchone()
    return row[0] if row else None


def get_bugs(run_id, robust_only=True, limit=10, offset=0):
    q = "SELECT %s FROM bugs WHERE run_id=%%s %s ORDER BY robust DESC, score DESC, n DESC LIMIT %%s OFFSET %%s" % (
        ",".join(COLS), "AND robust" if robust_only else "")
    with _c() as c:
        rows = c.execute(q, (run_id, limit, offset)).fetchall()
    return [dict(zip(COLS, r)) for r in rows]


def count_bugs(run_id, robust_only=True):
    with _c() as c:
        return c.execute("SELECT count(*) FROM bugs WHERE run_id=%s " + ("AND robust" if robust_only else ""),
                         (run_id,)).fetchone()[0]


def rule_keys(run_id):
    with _c() as c:
        rows = c.execute("SELECT pair, minute, side, mtg, hs, he, nloss FROM bugs WHERE run_id=%s AND robust", (run_id,)).fetchall()
    return {"|".join(str(x) for x in r) for r in rows}


# ---- settings
def kv_get(k):
    with _c() as c:
        row = c.execute("SELECT v FROM kv WHERE k=%s", (k,)).fetchone()
    return row[0] if row else None


def kv_set(k, v):
    with _c() as c:
        c.execute("INSERT INTO kv VALUES (%s,%s) ON CONFLICT (k) DO UPDATE SET v=EXCLUDED.v", (k, v))


# ---- live alerts
def alert_new(key, slot_ts, pair, side, mtg, rule):
    """True if this alert did not exist yet (so it must be sent), False if it was already sent."""
    with _c() as c:
        row = c.execute("INSERT INTO alerts (key, created, slot_ts, pair, side, mtg, state, step, rule) "
                        "VALUES (%s,%s,%s,%s,%s,%s,'open',NULL,%s) ON CONFLICT DO NOTHING RETURNING key",
                        (key, int(time.time()), slot_ts, pair, side, mtg, rule)).fetchone()
    return row is not None


def alerts_since(ts):
    with _c() as c:
        rows = c.execute("SELECT key, slot_ts, pair, side, mtg, rule, state, step FROM alerts WHERE slot_ts>=%s", (ts,)).fetchall()
    return [dict(key=r[0], slot_ts=r[1], pair=r[2], side=r[3], mtg=r[4], rule=r[5], state=r[6], step=r[7]) for r in rows]


def alerts_open():
    with _c() as c:
        rows = c.execute("SELECT key, slot_ts, pair, side, mtg, rule FROM alerts WHERE state='open' ORDER BY slot_ts").fetchall()
    return [dict(key=r[0], slot_ts=r[1], pair=r[2], side=r[3], mtg=r[4], rule=r[5]) for r in rows]


def alert_close(key, state, step):
    with _c() as c:
        c.execute("UPDATE alerts SET state=%s, step=%s WHERE key=%s", (state, step, key))
        c.execute("DELETE FROM alerts WHERE created < %s", (int(time.time()) - 30 * 86400,))


def tally(since_ts):
    """(wins, losses, unknown, open) of alerts whose slot is >= since_ts."""
    with _c() as c:
        rows = c.execute("SELECT state, count(*) FROM alerts WHERE slot_ts>=%s GROUP BY state", (since_ts,)).fetchall()
    d = dict(rows)
    return d.get("win", 0), d.get("loss", 0), d.get("unknown", 0), d.get("open", 0)

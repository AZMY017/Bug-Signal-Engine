"""READ-ONLY access to the live bot's candle databases (Neon). Nothing here ever writes."""
import os

import numpy as np
import psycopg

import pairs

URLS = [os.environ[k].strip() for k in sorted(os.environ) if k.startswith("DATABASE_URL_") and os.environ[k].strip()]

_SQL = ("SELECT ts, (CASE WHEN c>o THEN 1 WHEN c<o THEN -1 ELSE 0 END)::smallint FROM candles "
        "WHERE pair=%s AND ts>=%s AND ts<%s ORDER BY ts")


_SQL_OC = ("SELECT ts, o::float8, c::float8 FROM candles "
           "WHERE pair=%s AND ts>=%s AND ts<%s ORDER BY ts")


def load_oc(pair, from_ts, to_ts):
    """-> (ts int64, open float64, close float64) of the M1 candles in [from_ts, to_ts). Engine 3 needs open/close
    (not only the colour) because 3/5-minute candles are built from the one-minute ones."""
    rows_all = []
    with _conn(pair) as c:
        with c.cursor(name="e3load") as cur:
            cur.execute(_SQL_OC, (pairs.PID[pair], int(from_ts), int(to_ts)))
            while True:
                rows = cur.fetchmany(50000)
                if not rows:
                    break
                rows_all.append(np.array(rows, dtype=np.float64))
    if not rows_all:
        return np.zeros(0, np.int64), np.zeros(0), np.zeros(0)
    a = np.concatenate(rows_all)
    return a[:, 0].astype(np.int64), a[:, 1], a[:, 2]


def enabled():
    return bool(URLS)


def _conn(pair):
    c = psycopg.connect(URLS[pairs.shard(pair, len(URLS))], connect_timeout=30)
    c.read_only = True  # the database itself refuses any write from this connection
    return c


def load_dir(pair, from_ts, to_ts):
    """-> (ts int64 array, direction int8 array: +1 green, -1 red, 0 doji) for [from_ts, to_ts)."""
    ts_p, d_p = [], []
    with _conn(pair) as c:
        with c.cursor(name="bugscan") as cur:
            cur.execute(_SQL, (pairs.PID[pair], int(from_ts), int(to_ts)))
            while True:
                rows = cur.fetchmany(50000)
                if not rows:
                    break
                a = np.array(rows, dtype=np.int64)
                ts_p.append(a[:, 0])
                d_p.append(a[:, 1].astype(np.int8))
    if not ts_p:
        return np.zeros(0, np.int64), np.zeros(0, np.int8)
    return np.concatenate(ts_p), np.concatenate(d_p)


def bounds(pair):
    """(first_ts, last_ts) of the stored M1 candles, (None, None) if nothing."""
    with _conn(pair) as c:
        return c.execute("SELECT min(ts), max(ts) FROM candles WHERE pair=%s", (pairs.PID[pair],)).fetchone()


def last_ts(pair):
    with _conn(pair) as c:
        row = c.execute("SELECT max(ts) FROM candles WHERE pair=%s", (pairs.PID[pair],)).fetchone()
    return row[0]

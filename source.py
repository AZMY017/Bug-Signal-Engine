"""READ-ONLY access to the candle databases (Neon). Nothing here ever writes.
Real market: DATABASE_URL_1/2/3 (live bot).  OTC market: OTC_DATABASE_URL_1/2/3 (OTC-MARKET-FUTURE-SIGNAL-BOT).
A pair whose name ends with "_otc" is read from the OTC databases, every other pair from the real ones."""
import datetime as dt
import os

import numpy as np
import psycopg

import pairs

URLS = [os.environ[k].strip() for k in sorted(os.environ) if k.startswith("DATABASE_URL_") and os.environ[k].strip()]
# Same order rule as the OTC bot (sorted by name length, then name) so the first guess for a pair is its own database.
OTC_URLS = [os.environ[k].strip() for k in sorted(os.environ, key=lambda k: (len(k), k))
            if k.startswith("OTC_DATABASE_URL_") and os.environ[k].strip()]
OTC_OFF = 6 * 3600     # the OTC bot stores candle time as naive Dhaka (UTC+6) local time

_SQL = ("SELECT ts, (CASE WHEN c>o THEN 1 WHEN c<o THEN -1 ELSE 0 END)::smallint FROM candles "
        "WHERE pair=%s AND ts>=%s AND ts<%s ORDER BY ts")


_SQL_OC = ("SELECT ts, o::float8, c::float8 FROM candles "
           "WHERE pair=%s AND ts>=%s AND ts<%s ORDER BY ts")


def load_oc(pair, from_ts, to_ts):
    """-> (ts int64, open float64, close float64) of the M1 candles in [from_ts, to_ts). Engine 3 needs open/close
    (not only the colour) because 3/5-minute candles are built from the one-minute ones."""
    if pairs.is_otc(pair):
        return _otc_load_oc(pair, from_ts, to_ts)
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
    if not URLS:
        raise RuntimeError("DATABASE_URL_1/2/3 সেট করা নেই (রিয়েল মার্কেটের ক্যান্ডেল ডেটাবেজ)।")
    c = psycopg.connect(URLS[pairs.shard(pair, len(URLS))], connect_timeout=30)
    c.read_only = True  # the database itself refuses any write from this connection
    return c


def load_dir(pair, from_ts, to_ts):
    """-> (ts int64 array, direction int8 array: +1 green, -1 red, 0 doji) for [from_ts, to_ts)."""
    if pairs.is_otc(pair):
        ts, o, c = _otc_load_oc(pair, from_ts, to_ts)
        return ts, np.sign(c - o).astype(np.int8)
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
    if pairs.is_otc(pair):
        return _otc_bounds(pair)
    with _conn(pair) as c:
        return c.execute("SELECT min(ts), max(ts) FROM candles WHERE pair=%s", (pairs.PID[pair],)).fetchone()


def last_ts(pair):
    if pairs.is_otc(pair):
        return _otc_bounds(pair)[1]
    with _conn(pair) as c:
        row = c.execute("SELECT max(ts) FROM candles WHERE pair=%s", (pairs.PID[pair],)).fetchone()
    return row[0]


# ---------------------------------------------------------------- OTC market
def otc_enabled():
    return bool(OTC_URLS)


def _otc_order(pair):
    """Database indexes to try: the one the OTC bot puts this pair on first, then the others (cheap: indexed by pair)."""
    n = len(OTC_URLS)
    guess = pairs.OTC_PAIRS.index(pair) % n if pair in pairs.OTC_PAIRS else 0
    return [guess] + [i for i in range(n) if i != guess]


def _otc_conn(i):
    c = psycopg.connect(OTC_URLS[i], connect_timeout=30)
    c.read_only = True  # the database itself refuses any write from this connection
    return c


def _naive(ts):
    """unix seconds (UTC) -> naive Dhaka local datetime, the way the OTC databases store time."""
    return dt.datetime.fromtimestamp(int(ts) + OTC_OFF, dt.timezone.utc).replace(tzinfo=None)


_OTC_SQL = ("SELECT (EXTRACT(EPOCH FROM ts) - %s)::float8, open::float8, close::float8 FROM candles "
            "WHERE pair=%s AND ts>=%s AND ts<%s ORDER BY ts")


def _otc_load_oc(pair, from_ts, to_ts):
    """-> (ts int64 unix UTC, open, close) of the stored OTC M1 candles in [from_ts, to_ts), same shape as load_oc()."""
    if not OTC_URLS:
        raise RuntimeError("OTC_DATABASE_URL_1/2/3 সেট করা নেই (OTC মার্কেটের ক্যান্ডেল ডেটাবেজ)।")
    for i in _otc_order(pair):
        rows_all = []
        with _otc_conn(i) as c:
            with c.cursor(name="otcload") as cur:
                cur.execute(_OTC_SQL, (OTC_OFF, pair, _naive(from_ts), _naive(to_ts)))
                while True:
                    rows = cur.fetchmany(50000)
                    if not rows:
                        break
                    rows_all.append(np.array(rows, dtype=np.float64))
        if rows_all:
            a = np.concatenate(rows_all)
            return a[:, 0].astype(np.int64), a[:, 1], a[:, 2]
    return np.zeros(0, np.int64), np.zeros(0), np.zeros(0)


def _otc_bounds(pair):
    if not OTC_URLS:
        raise RuntimeError("OTC_DATABASE_URL_1/2/3 সেট করা নেই (OTC মার্কেটের ক্যান্ডেল ডেটাবেজ)।")
    for i in _otc_order(pair):
        with _otc_conn(i) as c:
            lo, hi = c.execute("SELECT min(ts), max(ts) FROM candles WHERE pair=%s", (pair,)).fetchone()
        if lo is not None:
            f = lambda d: int(d.replace(tzinfo=dt.timezone.utc).timestamp()) - OTC_OFF
            return f(lo), f(hi)
    return None, None

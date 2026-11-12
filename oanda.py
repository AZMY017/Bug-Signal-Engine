"""Direct OANDA candles for the live alerts (only the few pairs/slots that matter, a handful of requests per hour).
Same key / same mid prices as the live bot, so a candle here equals the candle the live bot stores in Neon."""
import asyncio
import calendar
import logging
import os
import time

import httpx
import numpy as np

log = logging.getLogger("oanda")
KEY = os.environ.get("OANDA_KEY", "").strip()
HOSTS = {"practice": "https://api-fxpractice.oanda.com", "live": "https://api-fxtrade.oanda.com"}
_host = HOSTS.get(os.environ.get("OANDA_ENV", "").lower())
MIN_GAP = float(os.environ.get("OANDA_MIN_INTERVAL", "0.3"))
_lock = None
_next_ok = 0.0
_client = None
STATE = {"ok": 0.0, "limit_n": 0, "err": ""}


def enabled():
    return bool(KEY)


def _cl():
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=30)
    return _client


async def _throttle():
    global _lock, _next_ok
    if _lock is None:
        _lock = asyncio.Lock()
    async with _lock:
        wait = _next_ok - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        _next_ok = time.monotonic() + MIN_GAP


def _iso(ts):
    return time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z", time.gmtime(ts))


async def _get(pair, params):
    global _host, _next_ok
    hdr = {"Authorization": "Bearer " + KEY}
    last = None
    for h in ([_host] if _host else list(HOSTS.values())):
        for attempt in range(4):
            await _throttle()
            r = await _cl().get(h + "/v3/instruments/%s/candles" % pair, params=params, headers=hdr)
            if r.status_code == 429 or r.status_code >= 500:
                try:
                    wait = float(r.headers.get("Retry-After", ""))
                except ValueError:
                    wait = 3.0 * (attempt + 1)
                _next_ok = max(_next_ok, time.monotonic() + wait)
                STATE["limit_n"] += 1
                log.warning("OANDA HTTP %s for %s, waiting %.0fs", r.status_code, pair, wait)
                continue
            break
        else:
            raise RuntimeError("OANDA এখন রিকোয়েস্ট নিচ্ছে না (লিমিট/সার্ভার সমস্যা)")
        if r.status_code in (401, 403):
            last = r
            continue
        r.raise_for_status()
        _host = h
        STATE.update(ok=time.time(), err="")
        return r.json()
    STATE["err"] = "OANDA key কাজ করছে না (HTTP %s)" % (last.status_code if last else "?")
    raise RuntimeError(STATE["err"])


async def fetch_dirs(pair, from_ts, to_ts):
    """Completed M1 candles in [from_ts, to_ts) -> (ts int64 array, direction int8: +1 green, -1 red, 0 doji)."""
    data = await _get(pair, {"granularity": "M1", "price": "M", "from": _iso(from_ts), "to": _iso(to_ts)})
    ts, d = [], []
    for c in data.get("candles", []):
        if not c.get("complete"):
            continue
        t = calendar.timegm(time.strptime(c["time"][:19], "%Y-%m-%dT%H:%M:%S"))
        o, cl = float(c["mid"]["o"]), float(c["mid"]["c"])
        ts.append(t)
        d.append(1 if cl > o else (-1 if cl < o else 0))
    return np.array(ts, np.int64), np.array(d, np.int8)

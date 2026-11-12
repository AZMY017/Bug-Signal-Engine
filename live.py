"""Live alerts. Neon is NOT touched while this runs (so Neon's free compute hours are not eaten):
 - settings, watchlist and alert state live in memory (saved to the results DB only when something changes),
 - fresh candles come straight from OANDA, only for the pair of a bug whose slot starts within the next minute.
When a watched bug's slot is about to start, today's earlier signals of that bug are checked and the signal is sent
if the rule matches. After the candles close, the result (WIN / LOSS) is checked from OANDA and sent."""
import asyncio
import datetime as dt
import logging
import os
import time

import engine
import oanda
import pairs
import resultsdb as rdb

log = logging.getLogger("live")
LEAD = int(os.environ.get("ALERT_LEAD_SEC", "45"))        # alert goes out this many seconds before the slot candle opens
RES_SEC = int(os.environ.get("RESULT_CHECK_SEC", "8"))    # second of each minute when open alerts are checked
STALE = int(os.environ.get("MAX_DATA_AGE_SEC", "600"))    # newest OANDA candle older than this -> no alert (market closed)
GRACE = 150                                               # wait this long for a missing candle before calling a result "unknown"

DIAG = {"tick_at": 0.0, "last_alert": None, "skipped": "", "res_at": 0.0, "requests": 0}
_cfg = {"on": False, "chat": None, "top": 10}
_watch = {"dirty": True, "rules": []}
_alerts = {}   # key -> dict(key, slot_ts, pair, side, mtg, rule, state, step)


def rule_key(r):
    return "|".join(str(r[k]) for k in ("pair", "minute", "side", "mtg", "hs", "he", "nloss"))


def day_start(ts):
    loc = ts + 6 * 3600
    return loc // 86400 * 86400 - 6 * 3600


def hhmm(ts):
    return dt.datetime.fromtimestamp(ts, pairs.BDT).strftime("%H:%M")


def market_closed(t):
    """Forex weekend (UTC+6): Saturday 06:00 -> Monday 03:00."""
    n = dt.datetime.fromtimestamp(t, pairs.BDT)
    wd = n.weekday()
    return (wd == 5 and n.hour >= 6) or wd == 6 or (wd == 0 and n.hour < 3)


# ---------------------------------------------------------------- state (memory first, database only on change)
async def load_state():
    """Once at start-up (and after a restart): settings + alerts of the last 2 days."""
    on = await asyncio.to_thread(rdb.kv_get, "live_on")
    chat = await asyncio.to_thread(rdb.kv_get, "live_chat")
    top = await asyncio.to_thread(rdb.kv_get, "live_top")
    _cfg.update(on=(on == "1"), chat=(int(chat) if chat else None), top=int(top or 10))
    rows = await asyncio.to_thread(rdb.alerts_since, day_start(time.time()) - 86400)
    for a in rows:
        _alerts[a["key"]] = a
    _watch["dirty"] = True


async def cfg(force=False):
    return _cfg


async def set_cfg(**kw):
    for k, v in kw.items():
        _cfg[k] = v
        await asyncio.to_thread(rdb.kv_set, "live_" + k, ("1" if v is True else "0" if v is False else str(v)))
    return _cfg


def invalidate():
    _watch["dirty"] = True


async def watchlist(force=False):
    """Top robust rules of the newest scan; variants of the same underlying bug are merged (best one stays)."""
    if force or _watch["dirty"]:
        run = await asyncio.to_thread(rdb.latest_run)
        rules = []
        if run:
            rows = await asyncio.to_thread(rdb.get_bugs, run[0], True, 600, 0)
            rules = engine.dedupe(rows)[:_cfg["top"]]
        _watch.update(dirty=False, rules=rules)
    return _watch["rules"]


def tally(since_ts):
    w = l = u = o = 0
    for a in _alerts.values():
        if a["slot_ts"] >= since_ts:
            st = a["state"]
            w += st == "win"
            l += st == "loss"
            u += st == "unknown"
            o += st == "open"
    return w, l, u, o


# ---------------------------------------------------------------- messages
def fmt_alert(r, slot_ts):
    side = "UP (Call) 📈" if r["side"] == "UP" else "DOWN (Put) 📉"
    why = "সরাসরি সিগন্যাল" if r["nloss"] == 0 else "আজ আগের %d টা সিগন্যাল পরপর লস → শর্ত মিলেছে" % r["nloss"]
    mtg = "নেই" if r["mtg"] == 0 else "সর্বোচ্চ %d ধাপ (পরের %d ক্যান্ডেল)" % (r["mtg"], r["mtg"])
    te = ""
    if r["nte"]:
        te = " • টেস্ট পিরিয়ডে %d%% (%d টা)" % (round(100 * r["wte"] / r["nte"]), r["nte"])
    return ("🐞 বাগ সিগন্যাল\n"
            "💱 %s   ⏰ %s (UTC+6)\n"
            "%s\n"
            "🔁 MTG: %s\n"
            "📊 ইতিহাস: %d%% (%d/%d)%s\n"
            "ℹ️ %s\n"
            "⚠️ সর্বোচ্চ ব্যাক-টু-ব্যাক লস (ইতিহাসে): %d" % (
                pairs.pretty(r["pair"]), hhmm(slot_ts), side, mtg,
                round(100 * r["w"] / r["n"]), r["w"], r["n"], te, why, r["maxl"]))


# ---------------------------------------------------------------- alert + result ticks
async def alert_tick(send):
    """Runs ~LEAD seconds before every minute boundary."""
    if not _cfg["on"] or not _cfg["chat"]:
        return
    now = time.time()
    DIAG["tick_at"] = now
    slot = int(now // 60 + 1) * 60
    loc = slot + 6 * 3600
    hour, minute = (loc % 86400) // 3600, (loc % 3600) // 60
    rules = [r for r in await watchlist() if r["minute"] == minute and r["hs"] <= hour < r["he"]]
    if not rules:
        return                      # nothing to do: no request to OANDA, no database touched
    if market_closed(now):
        DIAG["skipped"] = "মার্কেট বন্ধ (সপ্তাহান্ত)"
        return
    cache = {}
    for r in rules:
        name = pairs.pretty(r["pair"])
        try:
            if r["pair"] not in cache:
                cache[r["pair"]] = await oanda.fetch_dirs(r["pair"], day_start(slot), slot)
                DIAG["requests"] += 1
            ts, d = cache[r["pair"]]
            if len(ts) == 0 or ts[-1] < slot - STALE:
                DIAG["skipped"] = "%s: OANDA-র ক্যান্ডেল পুরনো/মার্কেট বন্ধ" % name
                continue
            run_len = engine.live_run(ts, d, 1 if r["side"] == "UP" else -1, r["mtg"], r["minute"], r["hs"], hour)
            if run_len is None:
                DIAG["skipped"] = "%s: আজকের আগের সিগন্যাল জানা যায়নি" % name
                continue
            if r["nloss"] and run_len != r["nloss"]:
                continue
            key = "%s|%d" % (rule_key(r), slot)
            if key in _alerts:
                continue
            _alerts[key] = dict(key=key, slot_ts=slot, pair=r["pair"], side=r["side"], mtg=r["mtg"],
                                rule=rule_key(r), state="open", step=None)
            await send(_cfg["chat"], fmt_alert(r, slot))
            DIAG["last_alert"] = (time.time(), name, hhmm(slot))
            try:
                await asyncio.to_thread(rdb.alert_new, key, slot, r["pair"], r["side"], r["mtg"], rule_key(r))
            except Exception:  # noqa
                log.exception("alert save failed (the alert itself was sent)")
        except Exception as e:  # noqa
            DIAG["skipped"] = "%s: %s" % (name, e)
            log.exception("alert for %s failed", name)


async def result_tick(send):
    opens = [a for a in _alerts.values() if a["state"] == "open"]
    DIAG["res_at"] = time.time()
    now = time.time()
    for a in opens:
        k = a["mtg"]
        end = a["slot_ts"] + (k + 1) * 60
        if now < end + 3:
            continue
        side = 1 if a["side"] == "UP" else -1
        try:
            ts, d = await oanda.fetch_dirs(a["pair"], a["slot_ts"], end)
            DIAG["requests"] += 1
        except Exception:  # noqa
            log.exception("result read failed")
            continue
        got = {int(t): int(x) for t, x in zip(ts, d)}
        state, step, wait = None, None, False
        for j in range(k + 1):
            t = a["slot_ts"] + 60 * j
            if t not in got:
                if now < t + 60 + GRACE:
                    wait = True
                else:
                    state = "unknown"
                break
            if got[t] == side:
                state, step = "win", j
                break
        else:
            state = "loss"
        if wait or state is None:
            continue
        a["state"], a["step"] = state, (step if step is not None else -1)
        try:
            await asyncio.to_thread(rdb.alert_close, a["key"], state, a["step"])
        except Exception:  # noqa
            log.exception("alert close save failed")
        if not _cfg["chat"]:
            continue
        w, l, u, o = tally(day_start(a["slot_ts"]))
        head = "%s | %s | %s" % (pairs.pretty(a["pair"]), hhmm(a["slot_ts"]), a["side"])
        if state == "win":
            txt = "✅ WIN (%s)\n%s" % ("Non-MTG" if step == 0 else "MTG%d" % step, head)
        elif state == "loss":
            txt = "❌ LOSS\n%s" % head
        else:
            txt = "❔ রেজাল্ট জানা গেল না (ক্যান্ডেল আসেনি)\n%s" % head
        txt += "\n📅 আজকের হিসাব: ✅ %d • ❌ %d%s" % (w, l, (" • ❔ %d" % u) if u else "")
        await send(_cfg["chat"], txt)


async def _sleep_to(sec_of_minute):
    now = time.time()
    t = int(now // 60) * 60 + sec_of_minute
    if t <= now + 0.5:
        t += 60
    await asyncio.sleep(t - now)


async def alert_loop(send):
    while True:
        await _sleep_to((60 - LEAD) % 60)
        try:
            await alert_tick(send)
        except Exception:  # noqa
            log.exception("alert tick")


async def result_loop(send):
    while True:
        await _sleep_to(RES_SEC)
        try:
            await result_tick(send)
        except Exception:  # noqa
            log.exception("result tick")

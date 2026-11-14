"""Engine 10 — Signal Radar (live alerts built from the Engine 9 back-test).

1) build():      for the chosen pairs run the Engine 9 back-test (Engine 8's lowest-B2B minutes -> Engine 4's "signal after N
                 back-to-back losses"). A slot (pair, colour, minute) goes on the radar list only when its win ratio is 100%
                 with at least `minsig` signals. If several MTG levels were chosen, the lowest MTG that qualifies is kept.
2) alert_tick(): ~LEAD seconds before every minute, for each radar slot that starts now, today's real candles are read from
                 OANDA (Neon is NOT touched) and today's hours 12:00.. of that minute are replayed with the SAME code as the
                 back-test (engine4.outcomes). If the losing streak is exactly N (or N+ in "N or more" mode) -> signal is sent
                 (green slot = UP, red slot = DOWN).
3) result_tick(): after the candles close, WIN / LOSS is sent with today's tally.
Settings, the radar list and the alerts live in memory and are saved to the results DB (kv) only when something changes."""
import asyncio
import datetime as dt
import json
import logging
import time

import numpy as np

import engine3
import engine4
import engine9
import live
import oanda
import pairs
import resultsdb as rdb
import source

log = logging.getLogger("radar")
LEAD, RES_SEC, STALE, GRACE = live.LEAD, live.RES_SEC, live.STALE, live.GRACE
REFRESH_HM = (11, 30)                  # BDT: the list is rebuilt every day from this time (before the 12:00 trading window)
HOURS = engine3.HOURS
UP, DOWN = "UP (Call) 📈", "DOWN (Put) 📉"

DIAG = {"tick_at": 0.0, "last_alert": None, "skipped": "", "requests": 0, "res_at": 0.0, "refresh_err": "", "try_at": 0.0}
_cfg = {"on": False, "chat": None}
_state = {"rules": [], "meta": {}, "built_at": 0}
_alerts = {}      # key -> dict(key, slot_ts, pair, side, mtg, tf, state, step)


# ---------------------------------------------------------------- persistence
def _save(key, obj):
    if rdb.enabled():
        rdb.kv_set(key, json.dumps(obj, ensure_ascii=False))


def _load(key):
    if not rdb.enabled():
        return None
    v = rdb.kv_get(key)
    return json.loads(v) if v else None


async def load_state():
    cfg = await asyncio.to_thread(_load, "radar_cfg")
    if cfg:
        _cfg.update(on=bool(cfg.get("on")), chat=cfg.get("chat"))
    st = await asyncio.to_thread(_load, "radar_rules")
    if st:
        _state.update(rules=st.get("rules", []), meta=st.get("meta", {}), built_at=st.get("built_at", 0))
    al = await asyncio.to_thread(_load, "radar_alerts") or []
    lim = live.day_start(time.time()) - 86400
    for a in al:
        if a["slot_ts"] >= lim:
            _alerts[a["key"]] = a


def cfg():
    return _cfg


async def set_cfg(**kw):
    _cfg.update(kw)
    await asyncio.to_thread(_save, "radar_cfg", _cfg)


def state():
    return _state


async def set_rules(rules, meta):
    _state.update(rules=rules, meta=meta, built_at=int(time.time()))
    await asyncio.to_thread(_save, "radar_rules", _state)


def _save_alerts():
    lim = live.day_start(time.time()) - 86400
    _save("radar_alerts", [a for a in _alerts.values() if a["slot_ts"] >= lim])


def tally(since_ts):
    w = l = u = o = 0
    for a in _alerts.values():
        if a["slot_ts"] >= since_ts:
            w += a["state"] == "win"
            l += a["state"] == "loss"
            u += a["state"] == "unknown"
            o += a["state"] == "open"
    return w, l, u, o


# ---------------------------------------------------------------- 1) build the radar list
def rules_from(res, minsig):
    """Engine 9 result of one pair -> radar rules (100% win ratio and >= minsig signals)."""
    best = {}
    for k in res["ks"]:                                   # lowest MTG first -> it wins when several qualify
        for side in ("G", "R"):
            for r in res["per"][k][side]["rows"]:
                if r["sig"] >= minsig and r["win"] == r["sig"]:
                    best.setdefault((r["minute"], side), dict(
                        pair=res["pair"], minute=r["minute"], side=side, mtg=k, tf=res["tf"], n=res["n_loss"],
                        atleast=bool(res["atleast"]), sig=r["sig"], win=r["win"], doji=r["doji"]))
    return sorted(best.values(), key=lambda x: (x["side"] != "G", x["minute"]))


async def build(st, progress=None):
    """st: tf, months, mtg (list), n, mode ('exact'|'atleast'), minsig, pairs. Caller holds the scan lock.
    -> list of dict(pair, days, rules) or None entries (pair without data) in the order of st['pairs']."""
    days = engine3.period_days(st["months"])
    plist = list(st["pairs"])
    gate = asyncio.Semaphore(3)

    async def one(pair):
        async with gate:
            now = int(time.time())
            ts, o, c = await asyncio.to_thread(source.load_oc, pair, now - (days + 10) * 86400, now + 86400)
            res = await asyncio.to_thread(engine9.analyze, pair, ts, o, c, st["tf"], st["months"], sorted(st["mtg"]),
                                          st["n"], "day", st["mode"] == "atleast")
            del ts, o, c
            if res is None:
                return None
            return dict(pair=pair, days=len(res["days"]), rules=rules_from(res, st["minsig"]))

    tasks = [asyncio.create_task(one(p)) for p in plist]
    out = []
    try:
        for i, tk in enumerate(tasks):
            out.append(await tk)
            if progress:
                await progress(i + 1, len(tasks))
    except BaseException:
        for tk in tasks:
            tk.cancel()
        raise
    return out


def _m(r, labels):
    return ":%02d %d/%d%s" % (r["minute"], r["win"], r["sig"], (" " + labels.get(r["mtg"], "MTG%d" % r["mtg"])) if labels.get("multi") else "")


def summary_text(per, st, labels):
    """Short text of a build result (one block per pair); returns list of messages <= 4000 chars."""
    lab = dict(labels, multi=len(st["mtg"]) > 1)
    head = "🎯 সিগন্যাল রাডার তালিকা\n%d মিনিট • %s • পরপর %d লস (%s) • কমপক্ষে %d সিগন্যাল, ১০০%% উইন" % (
        st["tf"], st["months_label"], st["n"], "N বা বেশি" if st["mode"] == "atleast" else "ঠিক N", st["minsig"])
    blocks, total = [], 0
    for p, e in zip(st["pairs"], per):
        name = pairs.pretty(p)
        if e is None:
            blocks.append("💱 %s — ডেটা নেই/কম" % name)
            continue
        g = [_m(r, lab) for r in e["rules"] if r["side"] == "G"]
        r_ = [_m(r, lab) for r in e["rules"] if r["side"] == "R"]
        total += len(e["rules"])
        lines = ["💱 %s (%d দিন)" % (name, e["days"])]
        lines.append("🟢 " + (" • ".join(g) if g else "—"))
        lines.append("🔴 " + (" • ".join(r_) if r_ else "—"))
        blocks.append("\n".join(lines))
    head += "\nমোট রাডার স্লট: %d" % total
    out, cur = [], head
    for b in blocks:
        if len(cur) + len(b) + 2 > 3900:
            out.append(cur)
            cur = head
        cur += "\n\n" + b
    out.append(cur)
    return out


def list_text(labels):
    rules, meta = _state["rules"], _state["meta"]
    if not rules:
        return "রাডার তালিকা এখনও খালি। আগে 🔄 তালিকা বানাও চাপো।"
    when = dt.datetime.fromtimestamp(_state["built_at"], pairs.BDT).strftime("%d %b %H:%M")
    lab = dict(labels, multi=len(set(r["mtg"] for r in rules)) > 1)
    lines = ["🎯 রাডার তালিকা (বানানো: %s)\n%d মিনিট • %s • পরপর %d লস • কমপক্ষে %d সিগন্যাল" % (
        when, meta.get("tf", 1), meta.get("months_label", ""), meta.get("n", 0), meta.get("minsig", 0))]
    by = {}
    for r in rules:
        by.setdefault(r["pair"], []).append(r)
    for p, rs in by.items():
        g = [_m(r, lab) for r in rs if r["side"] == "G"]
        r_ = [_m(r, lab) for r in rs if r["side"] == "R"]
        lines.append("\n💱 %s\n🟢 %s\n🔴 %s" % (pairs.pretty(p), " • ".join(g) or "—", " • ".join(r_) or "—"))
    return "\n".join(lines)[:4000]


# ---------------------------------------------------------------- 2) live check (same code as the back-test)
def streak_info(ts, o, c, r, hour):
    """Today's M1 candles (ts, o, c) -> (due, streak, loss_hours) for rule r at trading hour `hour` (BDT).
    due == the back-test would take a signal on this slot (exact N, or N or more)."""
    tf = r["tf"]
    lt, d = engine3.to_tf(ts, o, c, tf)
    if len(lt) == 0:
        return None
    _, dirm, val = engine3.build_grid(lt, d, tf)
    out, _, _ = engine4.outcomes(dirm, val, tf, r["minute"], np.array([0]), r["side"], r["mtg"])
    streak, hrs = 0, []
    for j in range(max(0, hour - HOURS[0])):
        v = int(out[0, j])
        if v == 0:
            continue                      # no candle -> that trade is skipped, same as in the back-test
        if v == -1:
            streak += 1
            hrs.append(HOURS[j])
        else:
            streak, hrs = 0, []
    due = streak >= r["n"] if r["atleast"] else streak == r["n"]
    return due, streak, hrs


def judge(ts, o, c, slot, k, tf, side, now):
    """Candles from the slot on -> (state 'win'|'loss'|'unknown'|None, step, wait). None + wait=True -> check again later."""
    lt, d = engine3.to_tf(ts, o, c, tf) if len(ts) else (np.zeros(0, np.int64), np.zeros(0, np.int8))
    got = {int(a): int(b) for a, b in zip(lt, d)}
    s = 1 if side == "G" else -1
    for j in range(k + 1):
        t = slot + engine3.OFF + j * tf * 60
        if t not in got:
            if now < slot + (j + 1) * tf * 60 + GRACE:
                return None, None, True
            return "unknown", -1, False
        if got[t] == s or got[t] == 0:    # doji counts as a win (same as Engine 4)
            return "win", j, False
    return "loss", -1, False


def fmt_alert(r, slot, streak, hrs, labels):
    side = UP if r["side"] == "G" else DOWN
    mtg = "নেই" if r["mtg"] == 0 else "সর্বোচ্চ %d ধাপ (পরের %d ক্যান্ডেল)" % (r["mtg"], r["mtg"])
    done = ", ".join("%02d:%02d" % (h, r["minute"]) for h in hrs[-max(r["n"], 1):])
    return ("🎯 সিগন্যাল রাডার\n"
            "💱 %s   ⏰ %s (UTC+6)\n"
            "%s\n"
            "⏱ টাইম ফ্রেম: %d মিনিট\n"
            "🔁 MTG: %s\n"
            "📊 ইতিহাস: %d/%d (১০০%%)\n"
            "ℹ️ আজ এই স্লটে পরপর %d লস: %s" % (
                pairs.pretty(r["pair"]), live.hhmm(slot), side, r["tf"], mtg, r["win"], r["sig"], streak, done))


async def alert_tick(send):
    if not _cfg["on"] or not _cfg["chat"] or not _state["rules"]:
        return
    now = time.time()
    DIAG["tick_at"] = now
    slot = int(now // 60 + 1) * 60
    loc = slot + engine3.OFF
    hour, minute = (loc % 86400) // 3600, (loc % 3600) // 60
    rules = [r for r in _state["rules"] if r["minute"] == minute and HOURS[0] <= hour <= HOURS[-1]]
    if not rules:
        return                            # nothing to do -> no request to OANDA
    if live.market_closed(now):
        DIAG["skipped"] = "মার্কেট বন্ধ (সপ্তাহান্ত)"
        return
    cache = {}
    for r in rules:
        name = pairs.pretty(r["pair"])
        try:
            if r["pair"] not in cache:
                cache[r["pair"]] = await oanda.fetch_oc(r["pair"], live.day_start(slot), slot)
                DIAG["requests"] += 1
            ts, o, c = cache[r["pair"]]
            if len(ts) == 0 or ts[-1] < slot - STALE:
                DIAG["skipped"] = "%s: OANDA-র ক্যান্ডেল পুরনো/মার্কেট বন্ধ" % name
                continue
            info = streak_info(ts, o, c, r, hour)
            if info is None or not info[0]:
                continue
            key = "%s|%d|%s|%d|%d" % (r["pair"], r["minute"], r["side"], r["tf"], slot)
            if key in _alerts:
                continue
            _alerts[key] = dict(key=key, slot_ts=slot, pair=r["pair"], side=r["side"], mtg=r["mtg"], tf=r["tf"],
                                state="open", step=None)
            await send(_cfg["chat"], fmt_alert(r, slot, info[1], info[2], {}))
            DIAG["last_alert"] = (time.time(), name, live.hhmm(slot))
            try:
                await asyncio.to_thread(_save_alerts)
            except Exception:  # noqa
                log.exception("radar alert save failed (the alert itself was sent)")
        except Exception as e:  # noqa
            DIAG["skipped"] = "%s: %s" % (name, e)
            log.exception("radar alert for %s failed", name)


# ---------------------------------------------------------------- 3) results
async def result_tick(send):
    opens = [a for a in _alerts.values() if a["state"] == "open"]
    DIAG["res_at"] = time.time()
    now = time.time()
    for a in opens:
        end = a["slot_ts"] + (a["mtg"] + 1) * a["tf"] * 60
        if now < end + 3:
            continue
        try:
            ts, o, c = await oanda.fetch_oc(a["pair"], a["slot_ts"], end)
            DIAG["requests"] += 1
        except Exception:  # noqa
            log.exception("radar result read failed")
            continue
        state, step, wait = judge(ts, o, c, a["slot_ts"], a["mtg"], a["tf"], a["side"], now)
        if wait or state is None:
            continue
        a["state"], a["step"] = state, step
        try:
            await asyncio.to_thread(_save_alerts)
        except Exception:  # noqa
            log.exception("radar close save failed")
        if not _cfg["chat"]:
            continue
        w, l, u, _ = tally(live.day_start(a["slot_ts"]))
        head = "%s | %s | %s" % (pairs.pretty(a["pair"]), live.hhmm(a["slot_ts"]), "UP" if a["side"] == "G" else "DOWN")
        if state == "win":
            txt = "✅ WIN (%s)\n%s" % ("Non-MTG" if step == 0 else "MTG%d" % step, head)
        elif state == "loss":
            txt = "❌ LOSS\n%s" % head
        else:
            txt = "❔ রেজাল্ট জানা গেল না (ক্যান্ডেল আসেনি)\n%s" % head
        txt += "\n📅 আজকের রাডার হিসাব: ✅ %d • ❌ %d%s" % (w, l, (" • ❔ %d" % u) if u else "")
        await send(_cfg["chat"], txt)


# ---------------------------------------------------------------- loops
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
            log.exception("radar alert tick")


async def result_loop(send):
    while True:
        await _sleep_to(RES_SEC)
        try:
            await result_tick(send)
        except Exception:  # noqa
            log.exception("radar result tick")


def _bdt_date(t):
    return dt.datetime.fromtimestamp(t, pairs.BDT).date()


async def refresh_loop(refresh_cb):
    """Once a day (from REFRESH_HM, BDT) rebuilds the radar list with the saved settings - only while the radar is ON."""
    while True:
        await asyncio.sleep(30)
        try:
            now = time.time()
            n = dt.datetime.fromtimestamp(now, pairs.BDT)
            if (not _cfg["on"] or not _state["meta"].get("pairs") or live.market_closed(now)
                    or (n.hour, n.minute) < REFRESH_HM or n.hour >= 23
                    or _bdt_date(_state["built_at"] or 0) == n.date() or now - DIAG["try_at"] < 600):
                continue
            DIAG["try_at"] = now
            await refresh_cb()
        except Exception as e:  # noqa
            DIAG["refresh_err"] = str(e)
            log.exception("radar refresh")

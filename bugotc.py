"""Bug OTC Checker — /bugotcchecker (own file; every engine and /otctest stay untouched, nothing is imported from them).

What it does: you forward (or paste) the day list that /engine14 (or /engine12) sent — the message with the date heading and the rows
    M1 USDEGP-OTC 03:28 PUT ✅
For every row the bot reads, from the candles stored in Neon (candles.payout / open / close, read-only):
  * the PAYOUT % of that pair at the entry minute;
  * if payout >= the minimum (default 80 %)  -> the result is re-checked on the candles: ✅ (direct win), ✅¹ ✅² ✅³ (won at MTG step 1/2/3), ❌ (loss);
  * if payout <  the minimum                 -> NO result, the payout is written instead (e.g. 77%), and it is not counted as win/loss.
Then it sends the list back in the same style as the Quantex bot: date heading, the rows, Total / Win / Loss.

Rules (same as the engines): the pasted ✅/❌ are never trusted, the result is recomputed from the candles; a doji candle counts as a win;
a missing candle gives no result (❔). Times are BDT (UTC+6) = the way the OTC databases store time.
Settings (buttons of /bugotcchecker): MTG level (No-MTG / MTG1 / MTG2 / MTG3, default MTG1) and the minimum payout (default 80 %, any number by ✏️).
"""
import asyncio
import datetime as dt
import html
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from telegram import BotCommand, InlineKeyboardButton as Btn, InlineKeyboardMarkup as Markup
from telegram.error import BadRequest
from telegram.ext import ApplicationHandlerStop, CallbackQueryHandler, CommandHandler, MessageHandler, filters

import pairs
import source

log = logging.getLogger("bugotc")
BDT = pairs.BDT
NMIN = 26 * 60                  # minutes read per pair-day: the day + 2 hours (MTG steps of the last signals)
WAIT_SECS = 900                 # an unanswered ✏️ prompt expires after 15 minutes
ARM_SECS = 43200                # forwarded lists are read for 12 hours after /bugotcchecker (or until ⏹ / another engine is used)
PROMPT_SECS = 1.5               # after a burst of forwarded messages: wait this long, then ask 'finished?' (nothing is checked until the person presses ✅)
CACHE_TTL, CACHE_MAX = 600, 120
SUP = {1: "¹", 2: "²", 3: "³"}
MTG_LB = {0: "নন-MTG", 1: "MTG1", 2: "MTG2", 3: "MTG3"}
PAY_PRESETS = (70, 75, 80, 85)
D = {}                          # injected by register(): send, allowed, deny, lock, bot
S = dict(mtg=1, pay=80.0, wait=False, wait_at=0.0, armed=False, armed_at=0.0, last_date=None, days={}, buf=[], gen=0, task=None, prompt=None)
_LOCK = []                      # one asyncio.Lock (created on first use): two batches are never processed at the same time
_CACHE = {}                     # (pair, date) -> (stored_at, open[], close[], payout[])


# ---------------------------------------------------------------- parsing
LINE_RE = re.compile(r"^\s*M(\d{1,2})\s+([A-Za-z0-9]+)\s*[-_]\s*OTC\s+(\d{1,2}):(\d{2})(?!\d)\s*(BUY|PUT|UP|DOWN|CALL)?", re.I)
DATE_YMD = re.compile(r"(?<!\d)(\d{4})\s*[-./]\s*(\d{1,2})\s*[-./]\s*(\d{1,2})(?!\d)")
DATE_DMY = re.compile(r"(?<!\d)(\d{1,2})\s*[-./]\s*(\d{1,2})\s*[-./]\s*(\d{4})(?!\d)")
BN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def find_date(text):
    cands = []
    for m in DATE_YMD.finditer(text):
        cands.append((m.start(), (int(m.group(1)), int(m.group(2)), int(m.group(3)))))
    for m in DATE_DMY.finditer(text):
        cands.append((m.start(), (int(m.group(3)), int(m.group(2)), int(m.group(1)))))
    for _p, (y, mo, d) in sorted(cands):
        try:
            return dt.date(y, mo, d)
        except ValueError:
            continue
    return None


def parse(text):
    """-> dict(date or None, items=[dict(tf, pair, name, E (minute of day), side BUY|PUT)], skipped_pair, skipped_side)."""
    text = (text or "").replace("\ufe0f", "").translate(BN)
    items, seen, sk_pair, sk_side = [], set(), 0, 0
    for ln in text.splitlines():
        m = LINE_RE.match(ln)
        if not m:
            continue
        tf, name, hh, mi, side = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4)), (m.group(5) or "").upper()
        if not (1 <= tf <= 30 and 60 % tf == 0 and 0 <= hh < 24 and 0 <= mi < 60):
            continue
        pair = pairs.norm(name + "-OTC")
        if not pair or not pairs.is_otc(pair):
            sk_pair += 1
            continue
        side = {"UP": "BUY", "CALL": "BUY", "DOWN": "PUT"}.get(side, side)
        if side not in ("BUY", "PUT"):
            sk_side += 1
            continue
        key = (tf, pair, hh * 60 + mi, side)
        if key in seen:
            continue
        seen.add(key)
        items.append(dict(tf=tf, pair=pair, E=hh * 60 + mi, side=side, key=key))
    return dict(date=find_date(text), items=items, skipped_pair=sk_pair, skipped_side=sk_side)


def parse_days(text, carry=None):
    """Multi-day parser. -> dict(segs=[[date|None, [items]], ...], skipped_pair, skipped_side).
    A line with a date (e.g. '📅 - 2026.11.14') switches the current date; rows that come before any date in this text
    (a continuation like '(2/2)' without a date) use `carry` = the date of the previous message."""
    text = (text or "").replace("\ufe0f", "").translate(BN)
    cur, segs, seen, sk_pair, sk_side = carry, [], set(), 0, 0
    for ln in text.splitlines():
        m = LINE_RE.match(ln)
        if not m:
            d = find_date(ln)
            if d:
                cur = d
            continue
        tf, name, hh, mi, side = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4)), (m.group(5) or "").upper()
        if not (1 <= tf <= 30 and 60 % tf == 0 and 0 <= hh < 24 and 0 <= mi < 60):
            continue
        pair = pairs.norm(name + "-OTC")
        if not pair or not pairs.is_otc(pair):
            sk_pair += 1
            continue
        side = {"UP": "BUY", "CALL": "BUY", "DOWN": "PUT"}.get(side, side)
        if side not in ("BUY", "PUT"):
            sk_side += 1
            continue
        key = (tf, pair, hh * 60 + mi, side)
        if (cur, key) in seen:
            continue
        seen.add((cur, key))
        if not segs or segs[-1][0] != cur:
            segs.append([cur, []])
        segs[-1][1].append(dict(tf=tf, pair=pair, E=hh * 60 + mi, side=side, key=key))
    return dict(segs=segs, skipped_pair=sk_pair, skipped_side=sk_side)


# ---------------------------------------------------------------- database (read-only): candles.payout / open / close
def _window(d):
    lo = dt.datetime.combine(d, dt.time(0, 0))      # the OTC databases store naive Dhaka (UTC+6) local time
    return lo, lo + dt.timedelta(minutes=NMIN)


def _to_arrays(rows, lo):
    o, c, p = (np.full(NMIN, np.nan) for _ in range(3))
    for ts, op, cl, pay in rows:
        i = int((ts - lo).total_seconds() // 60)
        if 0 <= i < NMIN:
            o[i] = np.nan if op is None else op
            c[i] = np.nan if cl is None else cl
            p[i] = np.nan if pay is None else pay
    return o, c, p


_SQL = "SELECT ts, open::float8, close::float8, payout::float8 FROM candles WHERE pair=%s AND ts>=%s AND ts<%s ORDER BY ts"


def _fetch_group(i, keys):
    """One reused connection to database i -> {(pair, date): arrays or None when that database has no rows for it}."""
    out = {}
    with source._otc_conn(i) as con:
        for pair, d in keys:
            lo, hi = _window(d)
            rows = con.execute(_SQL, (pair, lo, hi)).fetchall()
            out[(pair, d)] = _to_arrays(rows, lo) if rows else None
    return out


def load_all(keys):
    """keys = set of (pair, date) -> ({key: (open[], close[], payout[]) or None}, error text). A few databases are read in parallel."""
    res, err = {}, ""
    now = time.time()
    todo = []
    for k in keys:
        e = _CACHE.get(k)
        if e and now - e[0] < CACHE_TTL:
            res[k] = e[1:]
        else:
            todo.append(k)
    if not todo:
        return res, err
    n = len(source.OTC_URLS)
    if n == 0:
        return res, "OTC_DATABASE_URL_* সেট করা নেই"
    groups = {}
    for k in todo:
        guess = pairs.OTC_PAIRS.index(k[0]) % n if k[0] in pairs.OTC_PAIRS else 0
        groups.setdefault(guess, []).append(k)
    got = {}
    with ThreadPoolExecutor(max_workers=max(1, min(3, len(groups)))) as ex:
        futs = {ex.submit(_fetch_group, i, ks): i for i, ks in groups.items()}
        for f, i in futs.items():
            try:
                got.update(f.result())
            except Exception as e:  # noqa
                log.exception("bugotc load failed (db %d)", i)
                err = str(e)[:120]
    for k in todo:                                   # a pair that is not on its guessed database: try the others (cheap, indexed by pair)
        if got.get(k) is None:
            guess = pairs.OTC_PAIRS.index(k[0]) % n if k[0] in pairs.OTC_PAIRS else 0
            for i in [x for x in range(n) if x != guess]:
                try:
                    r = _fetch_group(i, [k]).get(k)
                except Exception as e:  # noqa
                    err = str(e)[:120]
                    continue
                if r is not None:
                    got[k] = r
                    break
    for k in todo:
        res[k] = got.get(k)
        if got.get(k) is not None:
            _CACHE[k] = (time.time(),) + got[k]
    while len(_CACHE) > CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)))
    return res, err


# ---------------------------------------------------------------- the check of ONE signal
def _pay_pct(x):
    if x is None or x != x:
        return None
    return x * 100.0 if x <= 1.5 else float(x)        # stored as % (77); a fraction (0.77) is converted


def evaluate(it, arr, mtg, pay_min):
    """-> ('win', step) | ('loss', None) | ('low', payout %) | ('nopay', None) | ('miss', None)."""
    if arr is None:
        return "miss", None
    o, c, p = arr
    E, tf = it["E"], it["tf"]
    pay = _pay_pct(p[E]) if E < NMIN else None
    if pay is None:
        return "nopay", None
    if pay < pay_min:
        return "low", pay
    want = 1 if it["side"] == "BUY" else -1
    for k in range(mtg + 1):
        a = E + k * tf
        b = a + tf - 1
        if b >= NMIN or np.isnan(o[a:b + 1]).any() or np.isnan(c[a:b + 1]).any():
            return "miss", None                        # a candle is missing: no result is guessed
        d = np.sign(c[b] - o[a])
        if d == want or d == 0:                        # doji = win (same as the engines)
            return "win", k
    return "loss", None


def mark(kind, v):
    if kind == "win":
        return "✅" + SUP.get(v, "")
    if kind == "loss":
        return "❌"
    if kind == "low":
        return "%d%%" % int(round(v))
    return "❔"


def counts(vals):
    n = dict(win=0, loss=0, low=0, nopay=0, miss=0, w=[0, 0, 0, 0])      # n["w"][k] = wins at MTG step k (0 = non-MTG / direct)
    for k, v in vals:
        n[k] += 1
        if k == "win" and v is not None and 0 <= v < 4:
            n["w"][v] += 1
    return n


def _split_lines(n):
    """Win split, one line per step that is active under the current MTG setting."""
    steps = list(range(S["mtg"] + 1))
    return ["   %s %s: %d" % ("└" if i == len(steps) - 1 else "├", MTG_LB[k], n["w"][k]) for i, k in enumerate(steps)]


def _split_inline(n):
    return " • ".join("%s %d" % (MTG_LB[k], n["w"][k]) for k in range(S["mtg"] + 1))


def _rate(w, l):
    return "%.0f%%" % (100.0 * w / (w + l)) if w + l else "-"


def build_messages(d, rows, width, st_line, day_all=None):
    """rows = [(it, kind, v)] of THIS forwarded message -> list of HTML messages (<= ~3300 chars of rows each)."""
    lines = ["M%d %s %02d:%02d %s %s" % (it["tf"], pairs.pretty(it["pair"]).ljust(width), it["E"] // 60, it["E"] % 60, it["side"], mark(k, v))
             for it, k, v in rows]
    head = "📅 - %s\n▬▬▬▬▬▬▬▬▬ • ▬▬▬▬▬▬▬▬▬\n" % d.strftime("%Y.%m.%d")
    parts, cur, size = [], [], 0
    for r in lines:
        if cur and size + len(r) + 1 > 3300:
            parts.append(cur)
            cur, size = [], 0
        cur.append(r)
        size += len(r) + 1
    if cur:
        parts.append(cur)
    n = counts([(k, v) for _it, k, v in rows])
    foot = ["▬▬▬▬▬▬▬▬▬ • ▬▬▬▬▬▬▬▬▬",
            "🏆 Total : %d" % len(rows), "✅ Win: %d" % n["win"]] + _split_lines(n) + ["❌ Loss: %d" % n["loss"],
            "📉 কম পেআউট (পার্সেন্ট দেখানো): %d" % n["low"],
            "📌 ট্রেড সিগন্যাল: %d (✅ %d + ❌ %d)" % (n["win"] + n["loss"], n["win"], n["loss"])]
    if n["nopay"] or n["miss"]:
        foot.append("❔ ডেটা/পেআউট নেই: %d" % (n["nopay"] + n["miss"]))
    foot.append("🎯 উইন রেট: %s" % _rate(n["win"], n["loss"]))
    if day_all is not None:
        a = counts(day_all)
        foot.append("📅 পুরো দিন (এ পর্যন্ত ফরওয়ার্ড): Total %d • 📌 ট্রেড সিগন্যাল %d • ✅ %d • ❌ %d • 📉 %d • %s" % (
            len(day_all), a["win"] + a["loss"], a["win"], a["loss"], a["low"], _rate(a["win"], a["loss"])))
        foot.append("   ✅ ভাগ: " + _split_inline(a))
    foot.append("⚙️ %s • পেআউট ন্যূনতম %s%%" % (st_line, ("%g" % S["pay"])))
    out = []
    for i, part in enumerate(parts):
        h = head if len(parts) == 1 else head + "(%d/%d)\n" % (i + 1, len(parts))
        body = h + "<pre>%s</pre>" % html.escape("\n".join(part), quote=False)
        if i == len(parts) - 1:
            body += "\n" + html.escape("\n".join(foot), quote=False)
        out.append(body)
    return out


def build_summary(per_day, st_line):
    """per_day = [(date, counts)] -> one HTML message: a block per day + the grand total (wins split by MTG step)."""
    tot = dict(win=0, loss=0, low=0, nopay=0, miss=0, w=[0, 0, 0, 0])
    lines = []
    for d, n in per_day:
        for k in ("win", "loss", "low", "nopay", "miss"):
            tot[k] += n[k]
        for i in range(4):
            tot["w"][i] += n["w"][i]
        lines.append("%s | মোট %d | 📌 ট্রেড %d | ✅ %d | ❌ %d | 📉 %d | %s" % (
            d.strftime("%m.%d"), sum(n[k] for k in ("win", "loss", "low", "nopay", "miss")), n["win"] + n["loss"], n["win"], n["loss"], n["low"], _rate(n["win"], n["loss"])))
        lines.append("      ✅ ভাগ: " + _split_inline(n))
    lines.append("▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬")
    lines.append("সব মিলিয়ে | মোট %d | 📌 ট্রেড %d | ✅ %d | ❌ %d | 📉 %d | %s" % (
        sum(tot[k] for k in ("win", "loss", "low", "nopay", "miss")), tot["win"] + tot["loss"], tot["win"], tot["loss"], tot["low"], _rate(tot["win"], tot["loss"])))
    lines.append("      ✅ ভাগ:")
    lines.extend(_split_lines(tot))
    if tot["nopay"] or tot["miss"]:
        lines.append("❔ ডেটা/পেআউট নেই: %d" % (tot["nopay"] + tot["miss"]))
    return "📊 সারসংক্ষেপ (%d দিন)\n<pre>%s</pre>\n⚙️ %s • পেআউট ন্যূনতম %s%%" % (
        len(per_day), html.escape("\n".join(lines), quote=False), html.escape(st_line), "%g" % S["pay"])


# ---------------------------------------------------------------- Telegram glue
def _tick(on, label):
    return ("✅ " if on else "") + label


def text_main():
    on = S["armed"] and time.time() - S["armed_at"] < ARM_SECS
    return ("🧾 বাগ OTC চেকার\n\n"
            "/engine14-এর দিনের মেসেজ (📅 তারিখসহ) এখানে ফরওয়ার্ড করো — একসাথে অনেক মেসেজ/অনেক দিন ফরওয়ার্ড বা পেস্ট করা যায় (তারিখ না থাকলে আগের মেসেজের তারিখ ধরা হয়)। পাঠানো শেষ হলে বট জিজ্ঞেস করবে — ✅ চাপলে সব একবারে হিসাব হবে। প্রতিটা সিগন্যালের ওই মিনিটের পেআউট %% ডাটাবেজ থেকে দেখে:\n"
            "• পেআউট ≥ %g%% → রেজাল্ট: ✅ সরাসরি, ✅¹ ✅² ✅³ MTG ধাপে উইন, ❌ লস\n"
            "• পেআউট < %g%% → রেজাল্টের জায়গায় পার্সেন্ট (যেমন 77%%), উইন/লসে গোনা হয় না\n\n"
            "MTG: %s • পেআউট ন্যূনতম: %g%%\n%s") % (
        S["pay"], S["pay"], MTG_LB[S["mtg"]], S["pay"],
        "🟢 ফরওয়ার্ডের অপেক্ষায় আছি (একের পর এক অনেক মেসেজ ফরওয়ার্ড করা যাবে)" if on else "⚪ বন্ধ — /bugotcchecker চাপলে চালু হবে")


def kb_main():
    pay_row = [Btn(_tick(S["pay"] == v, "%d%%" % v), callback_data="bo:pay:%d" % v) for v in PAY_PRESETS]
    custom = S["pay"] not in PAY_PRESETS
    pay_row.append(Btn(_tick(custom, ("✏️ %g%%" % S["pay"]) if custom else "✏️"), callback_data="bo:pay:custom"))
    return Markup([
        [Btn(_tick(S["mtg"] == v, lb), callback_data="bo:mtg:%d" % v) for v, lb in MTG_LB.items()],
        pay_row,
        [Btn("🧹 দিনের হিসাব মুছো", callback_data="bo:clr:0"), Btn("⏹ বন্ধ", callback_data="bo:off:0")]])


async def cmd_bugotc(update, ctx):
    if not D["allowed"](update):
        return await D["deny"](update)
    S["armed"], S["armed_at"], S["wait"] = True, time.time(), False
    await update.effective_message.reply_text(text_main(), reply_markup=kb_main())


async def on_cb(update, ctx):
    q = update.callback_query
    if not D["allowed"](update):
        return await q.answer("অনুমতি নেই", show_alert=True)
    _p, key, val = q.data.split(":")
    await q.answer()
    if key == "mtg":
        S["mtg"] = int(val)
    elif key == "pay" and val == "custom":
        S["wait"], S["wait_at"] = True, time.time()
        await q.message.reply_text("✏️ কত পার্সেন্টের নিচে হলে রেজাল্ট দেখাবে না সেটা লেখো (যেমন: 80 অথবা 85)। ১ থেকে 100।")
        return
    elif key == "pay":
        S["pay"], S["wait"] = float(val), False
    elif key == "go":
        if not S["buf"]:
            try:
                await q.edit_message_text("ℹ️ হিসাব করার মতো জমা করা লিস্ট নেই। লিস্ট ফরওয়ার্ড করো।")
            except BadRequest:
                pass
            return
        S["prompt"] = None
        try:
            await q.message.delete()
        except Exception:  # noqa
            pass
        S["task"] = asyncio.create_task(_run_go(q.message.chat_id))     # runs in the background so the bot keeps reading messages
        return
    elif key == "cancel":
        S["buf"].clear()
        S["gen"] += 1
        S["prompt"] = None
        try:
            await q.edit_message_text("🗑 জমা করা লিস্ট মুছে ফেলা হলো।")
        except BadRequest:
            pass
        return
    elif key == "clr":
        S["days"].clear()
        S["buf"].clear()
        S["gen"] += 1
        S["last_date"] = None
        await _drop_prompt()
    elif key == "off":
        S["armed"], S["wait"] = False, False
        S["buf"].clear()
        S["gen"] += 1
        await _drop_prompt()
    try:
        await q.edit_message_text(text_main(), reply_markup=kb_main())
    except BadRequest:
        pass


async def on_text(update, ctx):
    """Group -8: reads a forwarded/pasted day list while armed (or the typed number of a ✏️ prompt). Everything else passes through."""
    msg = update.effective_message
    if not msg or not msg.text:
        return
    now = time.time()
    p = parse(msg.text)
    if S["wait"] and p["items"]:
        S["wait"] = False
    if S["wait"]:
        if now - S["wait_at"] > WAIT_SECS:
            S["wait"] = False
        elif D["allowed"](update):
            t = msg.text.translate(BN).replace("%", "").strip()
            try:
                v = float(t)
            except ValueError:
                v = -1
            if not (1 <= v <= 100):
                await msg.reply_text("বুঝিনি। ১ থেকে ১০০-এর মধ্যে একটা সংখ্যা লেখো (যেমন: 80)।")
                raise ApplicationHandlerStop
            S["pay"], S["wait"] = v, False
            await msg.reply_text(text_main(), reply_markup=kb_main())
            raise ApplicationHandlerStop
    if not S["armed"] or now - S["armed_at"] > ARM_SECS or not D["allowed"](update):
        return
    pd = parse_days(msg.text, S["last_date"])
    if not any(items for _d, items in pd["segs"]):
        d = find_date(msg.text.translate(BN))
        if d and len(msg.text) < 60:                    # just a date typed/forwarded: remember it for the next lists
            S["last_date"] = d
            await msg.reply_text("📅 তারিখ %s ধরা হলো। এবার লিস্ট ফরওয়ার্ড করো।" % d.strftime("%Y.%m.%d"))
            raise ApplicationHandlerStop
        return
    S["armed_at"] = now
    if any(d is None for d, _items in pd["segs"]):
        await msg.reply_text("📅 এই মেসেজে তারিখ পাইনি। আগে তারিখসহ মেসেজ ফরওয়ার্ড করো, অথবা শুধু তারিখ লেখো (যেমন 2026-11-13), তারপর লিস্ট দাও।")
        raise ApplicationHandlerStop
    S["last_date"] = pd["segs"][-1][0]                  # a following message without a date continues this date
    S["buf"].append(pd)
    S["gen"] += 1
    S["task"] = asyncio.create_task(_show_prompt(S["gen"], update.effective_chat.id))
    raise ApplicationHandlerStop


def _pending():
    """-> (messages, sorted dates, unique signals) of the collected-but-not-yet-checked lists."""
    per = {}
    for b in S["buf"]:
        for d, items in b["segs"]:
            per.setdefault(d, set()).update(it["key"] for it in items)
    return len(S["buf"]), sorted(per), sum(len(v) for v in per.values())


async def _drop_prompt():
    pr, S["prompt"] = S["prompt"], None
    if pr:
        try:
            await D["bot"]().delete_message(*pr)
        except Exception:  # noqa
            pass


async def _show_prompt(gen, chat_id):
    """After the last forwarded message of a burst: ONE 'finished?' message (always at the bottom of the chat) with the buttons."""
    try:
        await asyncio.sleep(PROMPT_SECS)
        if gen != S["gen"] or not S["buf"]:
            return                                      # a newer message came (its own timer will ask) or the list was cleared
        n, dates, total = _pending()
        await _drop_prompt()
        ds = ", ".join(d.strftime("%m.%d") for d in dates[:8]) + (" …" if len(dates) > 8 else "")
        text = ("📥 জমা হয়েছে: %d টা মেসেজ • %d দিন (%s) • %d টা সিগন্যাল\n\n"
                "লিস্ট পাঠানো শেষ হয়েছে? শেষ হলে নিচের বাটন চাপো। না চাপলে ধরে নেব আরও পাঠাবে — জমা থাকবে।") % (n, len(dates), ds, total)
        kb = Markup([[Btn("✅ হ্যাঁ, শেষ — হিসাব করো", callback_data="bo:go:0")], [Btn("🗑 জমা মুছে ফেলো", callback_data="bo:cancel:0")]])
        m = await D["send"](chat_id, text, reply_markup=kb)
        if m is not None and gen == S["gen"]:
            S["prompt"] = (chat_id, m.message_id)
        elif m is not None:                             # raced with a newer message: remove this stale prompt
            try:
                await D["bot"]().delete_message(chat_id, m.message_id)
            except Exception:  # noqa
                pass
    except Exception:  # noqa
        log.exception("bugotc prompt failed")


async def _run_go(chat_id):
    try:
        if not _LOCK:
            _LOCK.append(asyncio.Lock())
        async with _LOCK[0]:
            await _process(chat_id)
    except Exception as e:  # noqa
        log.exception("bugotc go failed")
        await D["send"](chat_id, "❌ বাগ OTC চেকার হয়নি: %s" % e)


async def _process(chat_id):
    buf, S["buf"] = S["buf"], []
    if not buf:
        return
    per = {}                                            # date -> {key: item}  (same day in several messages = one list)
    for b in buf:
        for d, items in b["segs"]:
            day_items = per.setdefault(d, {})
            for it in items:
                day_items.setdefault(it["key"], it)
    dates = sorted(per)
    total = sum(len(v) for v in per.values())
    sk_pair = sum(b["skipped_pair"] for b in buf)
    sk_side = sum(b["skipped_side"] for b in buf)
    prog = await D["send"](chat_id, "⏳ %d টা মেসেজ • %d দিন • %d টা সিগন্যালের পেআউট ও ক্যান্ডেল দেখছি…" % (len(buf), len(dates), total))
    try:
        data, err = await asyncio.to_thread(load_all, {(it["pair"], d) for d in dates for it in per[d].values()})
        per_day = []
        for d in dates:
            its = list(per[d].values())
            rows = [(it,) + evaluate(it, data.get((it["pair"], d)), S["mtg"], S["pay"]) for it in its]
            day = S["days"].setdefault(d, {})
            for it, k, v in rows:
                day[it["key"]] = (k, v)
            width = max(len(pairs.pretty(it["pair"])) for it in its)
            day_all = [kv for kv in day.values()] if len(day) > len(rows) else None
            for t in build_messages(d, rows, width, MTG_LB[S["mtg"]], day_all):
                await D["send"](chat_id, t, parse_mode="HTML")
            per_day.append((d, counts([(k, v) for _it, k, v in rows])))
        if len(per_day) > 1:
            await D["send"](chat_id, build_summary(per_day, MTG_LB[S["mtg"]]), parse_mode="HTML")
        notes = []
        if sk_pair:
            notes.append("%d টা লাইনের পেয়ার চিনিনি" % sk_pair)
        if sk_side:
            notes.append("%d টা লাইনে BUY/PUT নেই" % sk_side)
        if err:
            notes.append("ডাটাবেজ সমস্যা: %s" % err)
        if notes:
            await D["send"](chat_id, "ℹ️ " + " • ".join(notes))
    except Exception as e:  # noqa
        log.exception("bugotc failed")
        await D["send"](chat_id, "❌ বাগ OTC চেকার হয়নি: %s" % e)
    finally:
        if prog:
            try:
                await prog.delete()
            except Exception:  # noqa
                pass


async def _clear_wait(update, ctx):
    """Another engine's command / button was used: forget a pending ✏️ prompt and stop reading forwarded lists (does not stop the other handler)."""
    S["wait"], S["armed"] = False, False
    S["buf"].clear()
    S["gen"] += 1
    await _drop_prompt()


def register(app, send, allowed, deny, lock, bot_getter):
    """Called once from bot.py main(): own command (/bugotcchecker), callbacks (bo:) and text handler — nothing existing is changed."""
    D.update(send=send, allowed=allowed, deny=deny, lock=lock, bot=bot_getter)
    app.add_handler(CommandHandler("bugotcchecker", cmd_bugotc))
    app.add_handler(CallbackQueryHandler(on_cb, pattern=r"^bo:"))
    app.add_handler(CommandHandler(["engine3", "engine4", "engine6", "engine8", "engine9", "engine10", "engine11", "engine12", "engine13", "engine14",
                                    "otctest", "realtest"], _clear_wait), group=-9)
    app.add_handler(CallbackQueryHandler(_clear_wait, pattern=r"^(e3|e4|e6|e8|e9|e10|e11|e12|e13|e14|ot|rt):"), group=-9)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text), group=-8)


COMMAND = BotCommand("bugotcchecker", "বাগ OTC চেকার: লিস্ট ফরওয়ার্ড করলে পেআউট % ও রেজাল্ট দেখায়")

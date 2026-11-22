"""MTG Shift — /mtgshift (own file; every engine, /bugotcchecker and /otctest stay untouched, nothing is imported from them).

Idea: you forward the day list of /engine14 (or /engine12). For every row the FIRST candle (the listed minute) is NOT traded:
  * the first candle wins (direct win in the list)      -> the row is dropped (not counted at all);
  * otherwise the trade starts at the NEXT candle (listed minute + timeframe):
        next candle wins        -> counted as direct win (non-MTG)   [= old MTG1 win]
        2nd next candle wins    -> counted as MTG1 win               [= old MTG2 win]
        (optional MTG2 = 3rd next candle, which the old list never reached)
        everything lost         -> loss
Result is always recomputed from the candles stored in Neon (the pasted ✅/❌ are never trusted); doji = win (same as the engines).
Payout % is read at the NEW entry minute; below the minimum -> the % is shown instead of a result (not counted as win/loss).
Settings (buttons): MTG level (default MTG1), minimum payout (default 80 %), list view (all signals / only trade signals).
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

log = logging.getLogger("mtgshift")
BDT = pairs.BDT
NMIN = 26 * 60                  # minutes read per pair-day: the day + 2 hours (MTG steps of the last signals)
WAIT_SECS = 900                 # an unanswered ✏️ prompt expires after 15 minutes
ARM_SECS = 43200                # forwarded lists are read for 12 hours after /mtgshift (or until ⏹ / another engine is used)
PROMPT_SECS = 1.5               # after a burst of forwarded messages: wait this long, then ask 'finished?' (nothing is checked until the person presses ✅)
CACHE_TTL, CACHE_MAX = 600, 120
SUP = {1: "¹", 2: "²"}
MTG_LB = {0: "নন-MTG", 1: "MTG1", 2: "MTG2"}
PAY_PRESETS = (70, 75, 80, 85)
VIEW_LB = {"all": "📋 সব সিগন্যাল দেখাও", "trade": "💹 শুধু ট্রেড সিগন্যাল"}


def _trade():
    """True = 'শুধু ট্রেড সিগন্যাল' view: low-payout (and no-data) rows are not listed and not counted; only ✅ / ❌ rows remain."""
    return S["view"] == "trade"
D = {}                          # injected by register(): send, allowed, deny, lock, bot
S = dict(mtg=1, pay=80.0, view="all", wait=False, wait_at=0.0, armed=False, armed_at=0.0, last_date=None, days={}, buf=[], gen=0, task=None, prompt=None)
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
                log.exception("mtgshift load failed (db %d)", i)
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
    """-> ('skip', None) first candle won (row dropped) | ('win', step) step 0 = next candle, 1 = the one after | ('loss', None)
    | ('low', payout %) | ('nopay', None) | ('miss', None).  The trade starts at E + tf (the first candle E is NOT traded)."""
    if arr is None:
        return "miss", None
    o, c, p = arr
    E, tf = it["E"], it["tf"]
    want = 1 if it["side"] == "BUY" else -1
    b0 = E + tf - 1
    if b0 >= NMIN or np.isnan(o[E:b0 + 1]).any() or np.isnan(c[E:b0 + 1]).any():
        return "miss", None
    d0 = np.sign(c[b0] - o[E])
    if d0 == want or d0 == 0:                          # the listed (first) candle already won: not part of this test
        return "skip", None
    E2 = E + tf                                        # the real entry
    pay = _pay_pct(p[E2]) if E2 < NMIN else None
    if pay is None:
        return "nopay", None
    if pay < pay_min:
        return "low", pay
    for k in range(mtg + 1):
        a = E2 + k * tf
        b = a + tf - 1
        if b >= NMIN or np.isnan(o[a:b + 1]).any() or np.isnan(c[a:b + 1]).any():
            return "miss", None
        d = np.sign(c[b] - o[a])
        if d == want or d == 0:
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
    n = dict(win=0, loss=0, low=0, nopay=0, miss=0, skip=0, w=[0, 0, 0, 0])      # n["w"][k] = wins at MTG step k (0 = non-MTG / direct)
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
    full = rows
    rows = [r for r in rows if r[1] != "skip"]
    if _trade():
        rows = [r for r in rows if r[1] in ("win", "loss")]
    width = max([len(pairs.pretty(it["pair"])) for it, _k, _v in rows] or [1])
    lines = ["M%d %s %02d:%02d %s %s (%02d:%02d)" % (it["tf"], pairs.pretty(it["pair"]).ljust(width), (it["E"] + it["tf"]) // 60 % 24, (it["E"] + it["tf"]) % 60, it["side"],
                                                      mark(k, v), it["E"] // 60, it["E"] % 60)
             for it, k, v in rows] or ["(কোনো ট্রেড সিগন্যাল নেই)"]
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
    n = counts([(k, v) for _it, k, v in full])
    if _trade():
        foot = ["▬▬▬▬▬▬▬▬▬ • ▬▬▬▬▬▬▬▬▬",
                "🏆 Total : %d" % (n["win"] + n["loss"]), "✅ Win: %d" % n["win"]] + _split_lines(n) + ["❌ Loss: %d" % n["loss"]]
    else:
        foot = ["▬▬▬▬▬▬▬▬▬ • ▬▬▬▬▬▬▬▬▬",
                "🏆 Total : %d" % len(rows), "✅ Win: %d" % n["win"]] + _split_lines(n) + ["❌ Loss: %d" % n["loss"],
                "📉 কম পেআউট (পার্সেন্ট দেখানো): %d" % n["low"],
                "ট্রেড সিগন্যাল: %d (✅ %d + ❌ %d)" % (n["win"] + n["loss"], n["win"], n["loss"])]
    foot.append("⏭ বাদ (প্রথম ক্যান্ডেলেই উইন): %d • লিস্টে মোট: %d" % (n["skip"], len(full)))
    if n["nopay"] or n["miss"]:
        foot.append("❔ ডেটা/পেআউট নেই: %d%s" % (n["nopay"] + n["miss"], " (লিস্টে দেখানো হয়নি)" if _trade() else ""))
    foot.append("🎯 উইন রেট: %s" % _rate(n["win"], n["loss"]))
    if day_all is not None:
        a = counts(day_all)
        if _trade():
            foot.append("📅 পুরো দিন (এ পর্যন্ত ফরওয়ার্ড): Total %d • ✅ %d • ❌ %d • %s" % (
                a["win"] + a["loss"], a["win"], a["loss"], _rate(a["win"], a["loss"])))
        else:
            foot.append("📅 পুরো দিন (এ পর্যন্ত ফরওয়ার্ড): Total %d • ট্রেড সিগন্যাল %d • ✅ %d • ❌ %d • 📉 %d • %s" % (
                len(day_all) - a["skip"], a["win"] + a["loss"], a["win"], a["loss"], a["low"], _rate(a["win"], a["loss"])))
        foot.append("   ✅ ভাগ: " + _split_inline(a))
    foot.append("⚙️ %s • পেআউট ন্যূনতম %s%%%s" % (st_line, ("%g" % S["pay"]), " • শুধু ট্রেড সিগন্যাল" if _trade() else ""))
    out = []
    for i, part in enumerate(parts):
        h = head if len(parts) == 1 else head + "(%d/%d)\n" % (i + 1, len(parts))
        body = h + "<pre>%s</pre>" % html.escape("\n".join(part), quote=False)
        if i == len(parts) - 1:
            body += "\n" + html.escape("\n".join(foot), quote=False)
        out.append(body)
    return out


def build_summary(per_day, st_line):
    """per_day = [(date, counts)] -> one HTML message: ONE table row per day + a grand-total row (wins split by MTG step)."""
    tot = dict(win=0, loss=0, low=0, nopay=0, miss=0, skip=0, w=[0, 0, 0, 0])
    one_month = len({(d.year, d.month) for d, _n in per_day}) == 1
    steps = list(range(S["mtg"] + 1))
    tr = _trade()
    heads = (["Day", "Trd", "W", "L", "Rate"] if tr else ["Day", "Tot", "Trd", "W", "L", "Low", "Rate"]) + ["N" if k == 0 else "M%d" % k for k in steps] + ["Skp"]
    rows = []
    for d, n in per_day:
        for k in ("win", "loss", "low", "nopay", "miss", "skip"):
            tot[k] += n[k]
        for i in range(4):
            tot["w"][i] += n["w"][i]
        dn = d.strftime("%d" if one_month else "%m.%d")
        if tr:
            rows.append([dn, n["win"] + n["loss"], n["win"], n["loss"], _rate(n["win"], n["loss"])] + [n["w"][k] for k in steps] + [n["skip"]])
        else:
            rows.append([dn, sum(n[k] for k in ("win", "loss", "low", "nopay", "miss")), n["win"] + n["loss"],
                         n["win"], n["loss"], n["low"], _rate(n["win"], n["loss"])] + [n["w"][k] for k in steps] + [n["skip"]])
    if tr:
        allrow = ["All", tot["win"] + tot["loss"], tot["win"], tot["loss"], _rate(tot["win"], tot["loss"])] + [tot["w"][k] for k in steps] + [tot["skip"]]
    else:
        allrow = ["All", sum(tot[k] for k in ("win", "loss", "low", "nopay", "miss")), tot["win"] + tot["loss"], tot["win"], tot["loss"], tot["low"],
                  _rate(tot["win"], tot["loss"])] + [tot["w"][k] for k in steps] + [tot["skip"]]
    wd = [max(len(str(r[i])) for r in rows + [allrow, heads]) for i in range(len(heads))]

    def fmt(r):
        return " ".join(str(v).rjust(wd[i]) for i, v in enumerate(r))
    head = fmt(heads)
    lines = [head] + [fmt(r) for r in rows] + ["-" * len(head), fmt(allrow)]
    legend = ("Trd=ট্রেড সিগন্যাল (W+L) • W=উইন • L=লস • N=নন-MTG • M1/M2=MTG ধাপে উইন • Skp=বাদ" if tr else
              "Tot=মোট • Trd=ট্রেড সিগন্যাল (W+L) • W=উইন • L=লস • Low=কম পেআউট • N=নন-MTG • M1/M2=MTG ধাপে উইন • Skp=প্রথম ক্যান্ডেলে উইন, বাদ")
    extra = ("\n❔ ডেটা/পেআউট নেই: %d" % (tot["nopay"] + tot["miss"])) if (tot["nopay"] or tot["miss"]) else ""
    return "📊 সারসংক্ষেপ (%d দিন)\n<pre>%s</pre>\n%s%s\n⚙️ %s • পেআউট ন্যূনতম %s%%%s" % (
        len(per_day), html.escape("\n".join(lines), quote=False), html.escape(legend), extra, html.escape(st_line), "%g" % S["pay"],
        " • শুধু ট্রেড সিগন্যাল" if tr else "")


# ---------------------------------------------------------------- Telegram glue
def _tick(on, label):
    return ("✅ " if on else "") + label


def text_main():
    """HTML message (parse_mode HTML): bold headings/values, coloured emoji bars, quote blocks."""
    on = S["armed"] and time.time() - S["armed_at"] < ARM_SECS
    return ("🔀 <b>MTG শিফট টেস্ট</b>\n"
            "🟩🟩🟩🟩🟩🟩🟩🟩🟩\n"
            "📥 <b>কীভাবে ব্যবহার করবে</b>\n"
            "<blockquote>১. /engine14-এর দিনের মেসেজ (📅 তারিখসহ) এখানে ফরওয়ার্ড করো — একসাথে অনেক মেসেজ বা অনেক দিন দেওয়া যায়।\n২. পাঠানো শেষ হলে বট জিজ্ঞেস করবে — <b>✅ চাপলে</b> সব একবারে হিসাব হবে।</blockquote>\n"
            "🟦🟦🟦🟦🟦🟦🟦🟦🟦\n"
            "📖 <b>নিয়ম</b> — লিস্টের সময়ের <b>প্রথম ক্যান্ডেলে ট্রেড ধরা হয় না</b>\n"
            "<blockquote>⏭ প্রথম ক্যান্ডেলেই উইন → সারিটা <b>বাদ</b>, হিসাবে আসে না\n"
            "▶️ নইলে পরের ক্যান্ডেলে ঢোকা হয়:\n"
            "✅ ওই ক্যান্ডেল উইন → <b>নন-MTG</b>\n"
            "✅¹ তার পরের ক্যান্ডেল উইন → <b>MTG1</b>\n"
            "❌ সব লস → <b>লস</b>\n"
            "💰 পেআউট দেখা হয় <b>নতুন এন্ট্রি মিনিটে</b>; ন্যূনতমের কম হলে রেজাল্টের জায়গায় পার্সেন্ট দেখায়</blockquote>\n"
            "🟧🟧🟧🟧🟧🟧🟧🟧🟧\n"
            "⚙️ <b>বর্তমান সেটিং</b>\n"
            "🔁 MTG: <b>%s</b>\n"
            "💰 পেআউট ন্যূনতম: <b>%g%%</b>\n"
            "📋 লিস্ট: <b>%s</b>\n"
            "🟩🟩🟩🟩🟩🟩🟩🟩🟩\n"
            "%s") % (MTG_LB[S["mtg"]], S["pay"], VIEW_LB[S["view"]].split(" ", 1)[1],
        "🟢 <b>ফরওয়ার্ডের অপেক্ষায় আছি</b>" if on else "⚪ <b>বন্ধ</b> — /mtgshift চাপলে চালু হবে")


def kb_main():
    pay_row = [Btn(_tick(S["pay"] == v, "%d%%" % v), callback_data="ms:pay:%d" % v) for v in PAY_PRESETS]
    custom = S["pay"] not in PAY_PRESETS
    pay_row.append(Btn(_tick(custom, ("✏️ %g%%" % S["pay"]) if custom else "✏️"), callback_data="ms:pay:custom"))
    return Markup([
        [Btn(_tick(S["mtg"] == v, lb), callback_data="ms:mtg:%d" % v) for v, lb in MTG_LB.items()],
        pay_row,
        [Btn(_tick(S["view"] == v, lb), callback_data="ms:view:%s" % v) for v, lb in VIEW_LB.items()],
        [Btn("🧹 দিনের হিসাব মুছো", callback_data="ms:clr:0"), Btn("⏹ বন্ধ", callback_data="ms:off:0")]])


async def cmd_mtgshift(update, ctx):
    if not D["allowed"](update):
        return await D["deny"](update)
    S["armed"], S["armed_at"], S["wait"] = True, time.time(), False
    await update.effective_message.reply_text(text_main(), reply_markup=kb_main(), parse_mode="HTML")


async def on_cb(update, ctx):
    q = update.callback_query
    if not D["allowed"](update):
        return await q.answer("অনুমতি নেই", show_alert=True)
    _p, key, val = q.data.split(":")
    await q.answer()
    if key == "mtg":
        S["mtg"] = int(val)
    elif key == "view":
        S["view"] = "trade" if val == "trade" else "all"
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
        await q.edit_message_text(text_main(), reply_markup=kb_main(), parse_mode="HTML")
    except BadRequest:
        pass


async def on_text(update, ctx):
    """Group -10 (own group: engine14 also used -7, and a group runs only its first matching handler): reads a forwarded/pasted day list while armed (or the typed number of a ✏️ prompt). Everything else passes through."""
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
            await msg.reply_text(text_main(), reply_markup=kb_main(), parse_mode="HTML")
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
        kb = Markup([[Btn("✅ হ্যাঁ, শেষ — হিসাব করো", callback_data="ms:go:0")], [Btn("🗑 জমা মুছে ফেলো", callback_data="ms:cancel:0")]])
        m = await D["send"](chat_id, text, reply_markup=kb)
        if m is not None and gen == S["gen"]:
            S["prompt"] = (chat_id, m.message_id)
        elif m is not None:                             # raced with a newer message: remove this stale prompt
            try:
                await D["bot"]().delete_message(chat_id, m.message_id)
            except Exception:  # noqa
                pass
    except Exception:  # noqa
        log.exception("mtgshift prompt failed")


async def _run_go(chat_id):
    try:
        if not _LOCK:
            _LOCK.append(asyncio.Lock())
        async with _LOCK[0]:
            await _process(chat_id)
    except Exception as e:  # noqa
        log.exception("mtgshift go failed")
        await D["send"](chat_id, "❌ MTG শিফট হয়নি: %s" % e)


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
            width = max(len(pairs.pretty(it["pair"])) for it in its)       # (trade view recomputes it from the shown rows)
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
        log.exception("mtgshift failed")
        await D["send"](chat_id, "❌ MTG শিফট হয়নি: %s" % e)
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
    """Called once from bot.py main(): own command (/mtgshift), callbacks (ms:) and text handler — nothing existing is changed."""
    D.update(send=send, allowed=allowed, deny=deny, lock=lock, bot=bot_getter)
    app.add_handler(CommandHandler("mtgshift", cmd_mtgshift))
    app.add_handler(CallbackQueryHandler(on_cb, pattern=r"^ms:"))
    app.add_handler(CommandHandler(["engine3", "engine4", "engine6", "engine8", "engine9", "engine10", "engine11", "engine12", "engine13", "engine14",
                                    "otctest", "realtest", "bugotcchecker"], _clear_wait), group=-11)
    app.add_handler(CallbackQueryHandler(_clear_wait, pattern=r"^(e3|e4|e6|e8|e9|e10|e11|e12|e13|e14|ot|rt|bo):"), group=-11)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text), group=-10)


COMMAND = BotCommand("mtgshift", "MTG শিফট: প্রথম ক্যান্ডেল বাদ দিয়ে পরের ক্যান্ডেল থেকে ঢুকলে রেজাল্ট")

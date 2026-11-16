"""/otctest (OTC) and /realtest (real market) — replay a saved /engine10 radar list on the stored candles of chosen day(s), like a live run (no live data needed).

The list is the TEXT of an /engine10 result that the user pastes into the chat, so the list is frozen (it can not be changed by
the candles of the test day). The bot reads tf, MTG, N back-to-back losses ("exactly N" / "N or more") and every pair's
green / red minutes from that text, asks for the day(s), reads the M1 candles of that day from Neon (read-only, via source.load_oc),
and replays every slot with the SAME code as the back-test (engine4.outcomes): each hour the slot's minute is traded, N losses in a
row (the streak restarts every day) -> the next hour's trade is a SIGNAL; its result is a direct win / MTG win / loss (doji = win).
Messages: signal, then result (in order of time), then a summary + CSV. Optional separate chats: env OTC_TEST_CHAT_ID (OTC pairs) and
REAL_TEST_CHAT_ID (real-market pairs); a market without its own chat sends to the chat where /otctest was started.

Nothing of the other engines is touched; this file only registers its own command, callbacks and a text handler.
"""
import asyncio
import csv
import datetime as dt
import html
import io
import logging
import os
import re
import time

import numpy as np
from telegram import BotCommand, InlineKeyboardButton as Btn, InlineKeyboardMarkup as Markup  # noqa: F401
from telegram.error import BadRequest
from telegram.ext import ApplicationHandlerStop, CallbackQueryHandler, CommandHandler, MessageHandler, filters

import engine3 as e3
import engine4
import pairs
import source

log = logging.getLogger("otctest")
BDT = pairs.BDT
OFF = e3.OFF
TEST_CHAT = {"otc": os.environ.get("OTC_TEST_CHAT_ID", "").strip(),      # optional channel / group for the OTC replay messages
             "real": os.environ.get("REAL_TEST_CHAT_ID", "").strip()}   # optional channel / group for the real-market replay messages
UP, DOWN = "UP (Call) 📈", "DOWN (Put) 📉"
MAX_DAYS = 7                                                          # most days in one test

def _new():
    return dict(step=None, text="", rules=None, hdr=None, mode="all", stop=False, running=False)     # step: None | "paste" | "date"


S = {"otc": _new(), "real": _new()}     # everything is kept apart per market: /otctest -> S["otc"], /realtest -> S["real"]
LAST = {"m": None}                      # market whose text input is awaited most recently
D = {}          # injected from bot.py: send, allowed, deny, lock
NAME = {"otc": "🧪 OTC", "real": "📈 রিয়েল"}
CMD = {"otc": "/otctest", "real": "/realtest"}
PFX = {"otc": "ot", "real": "rt"}


def _reset(m):
    S[m].update(step=None, text="", rules=None, hdr=None, stop=False)


def _digits(s):
    return (s or "").translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789"))


# ---------------------------------------------------------------- 1) read the pasted /engine10 text
def parse(text):
    """-> (hdr dict(tf, n, atleast, mtgs), rules list, error str or None).
    rule = dict(pair, side 'G'|'R', minute, mtg)."""
    text = _digits(text)
    cut = text.find("━")                        # the last block ("all pairs merged") has no pair names -> ignore it
    if cut >= 0:
        text = text[:cut]
    first = text.find("💱")
    head = text[:first] if first >= 0 else text
    m_tf = re.search(r"(\d+)\s*মিনিট", head)
    m_n = re.search(r"পরপর\s*(\d+)\s*লস", head)
    if not m_tf or not m_n:
        return None, [], "হেডার লাইনটা পাইনি (যেমন: 1 মিনিট • 25 দিন • MTG1 • পরপর 3 লস (ঠিক N) ...)। /engine10-এর পুরো টেক্সট, প্রথম মেসেজসহ, পাঠাও।"
    tf, n = int(m_tf.group(1)), int(m_n.group(1))
    atleast = "N বা বেশি" in head
    mtgs = sorted({int(x) for x in re.findall(r"MTG(\d)", head)} | ({0} if "নন-MTG" in head else set()))
    if not mtgs:
        return None, [], "হেডারে MTG লেখা পাইনি।"
    if tf <= 0 or 60 % tf:
        return None, [], "টাইম ফ্রেম বুঝিনি: %d" % tf
    rules, seen = [], set()
    cur = None
    for line in text[first:].split("\n") if first >= 0 else []:
        line = line.strip()
        if line.startswith("💱"):
            nm = line[1:].strip()
            nm = nm.split("—")[0].strip()
            cur = pairs.norm(nm)
            if cur is None:
                return None, [], "পেয়ারের নাম চিনিনি: %s" % nm
            continue
        if cur is None or not (line.startswith("🟢") or line.startswith("🔴")):
            continue
        side = "G" if line.startswith("🟢") else "R"
        for mm, k in re.findall(r":(\d{2})(?:·M(\d))?", line):
            minute = int(mm)
            if minute % tf:
                continue
            mt = int(k) if k else mtgs[0]
            key = (cur, side, minute)
            if key in seen:
                continue
            seen.add(key)
            rules.append(dict(pair=cur, side=side, minute=minute, mtg=mt))
    if not rules:
        return None, [], "কোনো স্লট পাইনি। /engine10-এর তালিকার টেক্সট (💱 পেয়ার, 🟢 / 🔴 মিনিট) পাঠাও।"
    return dict(tf=tf, n=n, atleast=atleast, mtgs=mtgs), rules, None


def summary_line(hdr, rules):
    ps = sorted({r["pair"] for r in rules})
    g = sum(1 for r in rules if r["side"] == "G")
    mt = "+".join("নন-MTG" if k == 0 else "MTG%d" % k for k in hdr["mtgs"])
    return "%d মিনিট • %s • পরপর %d লস (%s)\n%d পেয়ার • %d স্লট (🟢 %d, 🔴 %d)" % (
        hdr["tf"], mt, hdr["n"], "N বা বেশি" if hdr["atleast"] else "ঠিক N", len(ps), len(rules), g, len(rules) - g)


# ---------------------------------------------------------------- 2) dates
def parse_dates(text, today):
    """'17' '17,18,19' '17-19' '17/11' '2026-11-17' 'গতকাল' -> sorted list of dt.date (BDT), or None."""
    t = _digits(text).strip().lower()
    if t in ("গতকাল", "yesterday"):
        return [today - dt.timedelta(days=1)]
    if t in ("আজ", "today"):
        return [today]
    out = []
    for tok in re.split(r"[,\s]+", t):
        if not tok:
            continue
        m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", tok)
        r = re.fullmatch(r"(\d{1,2})-(\d{1,2})", tok)
        s = re.fullmatch(r"(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?", tok)
        d1 = re.fullmatch(r"(\d{1,2})", tok)
        try:
            if m:
                out.append(dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
            elif s:
                y = int(s.group(3)) if s.group(3) else today.year
                y = y + 2000 if y < 100 else y
                out.append(dt.date(y, int(s.group(2)), int(s.group(1))))
            elif r:
                a, b = int(r.group(1)), int(r.group(2))
                if not (1 <= a <= b <= 31):
                    return None
                for x in range(a, b + 1):
                    out.append(_day_of_month(x, today))
            elif d1:
                out.append(_day_of_month(int(d1.group(1)), today))
            else:
                return None
        except (ValueError, TypeError):
            return None
    return sorted(set(out)) or None


def _day_of_month(d, today):
    """Day number only -> that day of this month (or of last month when it is still in the future)."""
    try:
        c = dt.date(today.year, today.month, d)
    except ValueError:
        c = None
    if c is None or c > today:
        y, m = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
        c = dt.date(y, m, d)
    return c


def day_ts(d):
    """BDT calendar date -> unix seconds of its 00:00 (UTC+6)."""
    return int(dt.datetime(d.year, d.month, d.day, tzinfo=BDT).timestamp())


def _mk(pair):
    return "otc" if pairs.is_otc(pair) else "real"


def _dest(chat_id, market):
    """Where the replay messages of a market go: its own channel (env) or the chat that started the test."""
    v = TEST_CHAT.get(market) or chat_id
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


def split_market(sigs, info, rules, market):
    """Only the part of a replay that belongs to one market."""
    ss = [x for x in sigs if _mk(x["pair"]) == market]
    inf = dict(no_data=[x for x in info["no_data"] if _mk(x[0]) == market], partial=[x for x in info["partial"] if _mk(x[0]) == market],
               missing=info["missing_m"][market], missing_m=info["missing_m"])
    return ss, inf, [r for r in rules if _mk(r["pair"]) == market]


# ---------------------------------------------------------------- 3) replay (same code as the back-test)
def replay(rules, hdr, dates):
    """-> (signals list, info dict). signal = dict(ts, date, pair, side, minute, hour, mtg, res 1|-1, lvl, doji, streak, loss_hours).
    info: no_data[(pair, date)], partial[(pair, date, last 'HH:MM')], skipped_missing (int)."""
    tf, n, atleast = hdr["tf"], hdr["n"], hdr["atleast"]
    info = dict(no_data=[], partial=[], missing=0, missing_m={"otc": 0, "real": 0})
    sigs = []
    by_pair = {}
    for r in rules:
        by_pair.setdefault(r["pair"], []).append(r)
    lo = day_ts(dates[0])
    hi = day_ts(dates[-1]) + 86400
    for pair, rs in by_pair.items():
        hours = e3.hours_for(pair)
        ts, o, c = source.load_oc(pair, lo, hi + 2 * 3600)          # 2 more hours: MTG steps of the last candles
        if len(ts) == 0:
            info["no_data"] += [(pair, d) for d in dates]
            continue
        lt, dd = e3.to_tf(ts, o, c, tf)
        if len(lt) == 0:
            info["no_data"] += [(pair, d) for d in dates]
            continue
        d0, dirm, val = e3.build_grid(lt, dd, tf)
        nd = dirm.shape[0]
        for d in dates:
            row = (day_ts(d) + OFF) // 86400 - d0
            if row < 0 or row >= nd or not val[row].any():
                info["no_data"].append((pair, d))
                continue
            last_loc = int(lt[(lt // 86400) == (day_ts(d) + OFF) // 86400].max())
            last_hhmm = dt.datetime.fromtimestamp(last_loc - OFF, BDT).strftime("%H:%M")
            if last_loc < day_ts(d) + OFF + 23 * 3600:                      # candles end before the 23:00 hour -> day not complete
                info["partial"].append((pair, d, last_hhmm))
            for r in rs:
                out, lvl, dj = engine4.outcomes(dirm, val, tf, r["minute"], np.array([row]), r["side"], r["mtg"], hours)
                streak, loss_h = 0, []
                for j, h in enumerate(hours):
                    v = int(out[0, j])
                    due = streak >= n if atleast else streak == n
                    if v == 0:
                        if due:
                            info["missing"] += 1
                            info["missing_m"][_mk(pair)] += 1
                        continue
                    if due:
                        sigs.append(dict(ts=day_ts(d) + h * 3600 + r["minute"] * 60, date=d, pair=pair, side=r["side"], minute=r["minute"],
                                         hour=h, mtg=r["mtg"], res=v, lvl=int(lvl[0, j]), doji=bool(dj[0, j]), streak=streak,
                                         loss_hours=list(loss_h)))
                    if v == -1:
                        streak += 1
                        loss_h.append(h)
                    else:
                        streak, loss_h = 0, []
    sigs.sort(key=lambda x: (x["ts"], x["pair"], x["side"]))
    return sigs, info


# ---------------------------------------------------------------- 4) texts
def _mtg_txt(k):
    return "নেই" if k == 0 else "সর্বোচ্চ %d ধাপ (পরের %d ক্যান্ডেল)" % (k, k)


def fmt_signal(s, hdr):
    side = UP if s["side"] == "G" else DOWN
    done = ", ".join("%02d:%02d" % (h, s["minute"]) for h in s["loss_hours"][-max(hdr["n"], 1):])
    return ("%s টেস্ট (রিপ্লে) • %s\n💱 %s   ⏰ %02d:%02d (UTC+6)\n%s\n⏱ টাইম ফ্রেম: %d মিনিট\n🔁 MTG: %s\nℹ️ এই স্লটে পরপর %d লস: %s" % (
        "🧪 OTC" if pairs.is_otc(s["pair"]) else "📈 রিয়েল", s["date"].strftime("%d %b"), pairs.pretty(s["pair"]), s["hour"], s["minute"], side, hdr["tf"], _mtg_txt(s["mtg"]), s["streak"], done))


def fmt_result(s, tally):
    head = "%s | %02d:%02d | %s" % (pairs.pretty(s["pair"]), s["hour"], s["minute"], "UP" if s["side"] == "G" else "DOWN")
    if s["res"] == 1:
        t = "✅ WIN (%s%s)\n%s" % ("Non-MTG" if s["lvl"] == 0 else "MTG%d" % s["lvl"], ", ডোজি" if s["doji"] else "", head)
    else:
        t = "❌ LOSS\n%s" % head
    return t + "\n📅 এই দিনের হিসাব এ পর্যন্ত: ✅ %d • ❌ %d" % (tally[0], tally[1])


def stats(sigs):
    w = [s for s in sigs if s["res"] == 1]
    lv = {}
    for s in w:
        lv[s["lvl"]] = lv.get(s["lvl"], 0) + 1
    return len(sigs), len(w), len(sigs) - len(w), lv


def _lv_txt(lv):
    return " • ".join("%s %d" % ("সরাসরি উইন" if k == 0 else "MTG%d উইন" % k, lv[k]) for k in sorted(lv)) or "—"


SUP = {1: "¹", 2: "²", 3: "³"}


def _row_icon(s):
    if s["res"] != 1:
        return "❌"
    return "✅" + SUP.get(s["lvl"], "") if s["lvl"] else "✅"          # direct win = ✅ only, MTG1 win = ✅¹, MTG2 = ✅², MTG3 = ✅³


def _day_lists(sigs, hdr):
    """Per date: a header (📅 - 2026.11.16 + divider) and the day's signals in time order, one per row:
    'M5 USCrude-OTC 03:45 BUY ✅'. Rows are put into <pre> blocks (monospace) of <= 60 rows so a message never cuts a block in half."""
    width = max(len(pairs.pretty(s["pair"])) for s in sigs)
    out = []
    for d in sorted({s["date"] for s in sigs}):
        rows = ["M%d %s %02d:%02d %s %s" % (hdr["tf"], pairs.pretty(s["pair"]).ljust(width), s["hour"], s["minute"],
                                           ("BUY" if s["side"] == "G" else "PUT"), _row_icon(s))
                for s in sorted((x for x in sigs if x["date"] == d), key=lambda x: (x["hour"], x["minute"], x["pair"]))]
        out.append("📅 - %s\n▬▬▬▬▬▬▬▬▬ • ▬▬▬▬▬▬▬▬▬" % d.strftime("%Y.%m.%d"))
        for i in range(0, len(rows), 60):
            out.append(("PRE", "<pre>%s</pre>" % html.escape("\n".join(rows[i:i + 60]), quote=False)))
    return out


def summary_texts(sigs, info, hdr, dates, rules, partial_note=True):
    n, w, l, lv = stats(sigs)
    ds = ", ".join(d.strftime("%d %b") for d in dates)
    L = ["📊 %s টেস্ট রিপোর্ট • %s" % ("OTC" if rules and pairs.is_otc(rules[0]["pair"]) else "রিয়েল মার্কেট", ds), summary_line(hdr, rules), ""]
    if n == 0:
        L.append("এই দিন(গুলো)তে কোনো সিগন্যাল হয়নি।")
    else:
        L.append("মোট সিগন্যাল: %d\n✅ উইন %d (%s)\n❌ লস %d\n🎯 উইন রেশিও: %.1f%%" % (n, w, _lv_txt(lv), l, 100.0 * w / n))
    if len(dates) > 1 and n:
        L.append("")
        L.append("দিন ধরে:")
        for d in dates:
            ss = [s for s in sigs if s["date"] == d]
            a, b, c_, _ = stats(ss)
            L.append("  %s: %d সিগন্যাল • ✅ %d • ❌ %d%s" % (d.strftime("%d %b"), a, b, c_, (" • %.0f%%" % (100.0 * b / a)) if a else ""))
    if n:
        # per-day signal list (replaces the old "পেয়ার ধরে" block): date, then  M<tf> PAIR HH:MM ✅ / ✅¹ (MTG1 win) / ❌
        L.append("")
        L += _day_lists(sigs, hdr)
        bad = [s for s in sigs if s["res"] == -1]
        if bad:
            L.append("")
            L.append("❌ লসগুলো:")
            for s in bad:
                L.append("  %s %02d:%02d %s %s" % (s["date"].strftime("%d %b"), s["hour"], s["minute"], pairs.pretty(s["pair"]), "UP" if s["side"] == "G" else "DOWN"))
    notes = []
    if info["no_data"]:
        nd = sorted({d for _, d in info["no_data"]})
        notes.append("⚠️ ডেটা নেই: %d টা (পেয়ার/দিন), তারিখ: %s। ওই দিনের ক্যান্ডেল Neon-এ এখনো জমা হয়নি।" % (len(info["no_data"]), ", ".join(d.strftime("%d %b") for d in nd)))
    if partial_note and info["partial"]:
        notes.append("⚠️ অসম্পূর্ণ দিন: %d টা (পেয়ার/দিন), যেমন %s %s-এর ডেটা শেষ %s পর্যন্ত। বাকি ঘণ্টার সিগন্যাল ধরা হয়নি।" % (
            len(info["partial"]), pairs.pretty(info["partial"][0][0]), info["partial"][0][1].strftime("%d %b"), info["partial"][0][2]))
    if info["missing"]:
        notes.append("ℹ️ ক্যান্ডেল না থাকায় %d টা সিগন্যাল বাদ গেছে।" % info["missing"])
    if notes:
        L += [""] + notes
    # the messages are sent with parse_mode="HTML" (the signal rows sit in <pre> = monospace, so the columns line up)
    ents = [x[1] if isinstance(x, tuple) else html.escape(x, quote=False) for x in L]
    out, cur = [], ""
    for ent in ents:
        if cur and len(cur) + len(ent) + 1 > 3900:
            out.append(cur)
            cur = ""
        cur += ent + "\n"
    if cur.strip():
        out.append(cur)
    return out


def make_csv(sigs):
    bio = io.StringIO()
    w = csv.writer(bio)
    w.writerow(["date", "time_bdt", "pair", "direction", "result", "win_level", "doji_win", "losses_before"])
    for s in sigs:
        w.writerow([s["date"].isoformat(), "%02d:%02d" % (s["hour"], s["minute"]), pairs.pretty(s["pair"]), "UP" if s["side"] == "G" else "DOWN",
                    "WIN" if s["res"] == 1 else "LOSS", ("Non-MTG" if s["lvl"] == 0 else "MTG%d" % s["lvl"]) if s["res"] == 1 else "", int(s["doji"]), s["streak"]])
    return ("\ufeff" + bio.getvalue()).encode("utf-8")


# ---------------------------------------------------------------- 5) Telegram glue (one set per market)
def _kb_paste(m):
    p = PFX[m]
    return Markup([[Btn("✅ পাঠানো শেষ", callback_data="%s:done:0" % p), Btn("❌ বাতিল", callback_data="%s:cancel:0" % p)]])


def _kb_date(m):
    p = PFX[m]
    return Markup([[Btn("📨 মেসেজ: সব (লাইভের মতো)" if S[m]["mode"] == "all" else "📊 মেসেজ: শুধু সারাংশ", callback_data="%s:mode:0" % p)],
                   [Btn("গতকাল", callback_data="%s:day:y" % p), Btn("আজ", callback_data="%s:day:t" % p)],
                   [Btn("❌ বাতিল", callback_data="%s:cancel:0" % p)]])


async def _cmd(update, m):
    if not D["allowed"](update):
        return await D["deny"](update)
    st = S[m]
    if st["running"]:
        return await update.effective_message.reply_text("%s টেস্ট চলছে। শেষ হওয়া পর্যন্ত অপেক্ষা করো (বা মেসেজের ⏹ বাটন চাপো)।" % NAME[m])
    if m == "otc" and not source.otc_enabled():
        return await update.effective_message.reply_text("OTC_DATABASE_URL_* সেট করা নেই।")
    if m == "real" and not source.enabled():
        return await update.effective_message.reply_text("DATABASE_URL_* সেট করা নেই।")
    _reset(m)
    st["step"] = "paste"
    LAST["m"] = m
    await update.effective_message.reply_text(
        "%s টেস্ট (রিপ্লে)\n\n/engine10-এর %s তালিকার রেজাল্ট টেক্সট এখানে পাঠাও (কপি করে পেস্ট)। একাধিক মেসেজ হলে সবগুলো পাঠাও, তারপর ✅ চাপো।\n"
        "প্রথম মেসেজে হেডার লাইন (টাইম ফ্রেম, MTG, পরপর লস) থাকতে হবে।" % (NAME[m], "OTC" if m == "otc" else "রিয়েল মার্কেটের"), reply_markup=_kb_paste(m))


async def cmd_otctest(update, ctx):
    await _cmd(update, "otc")


async def cmd_realtest(update, ctx):
    await _cmd(update, "real")


async def on_text(update, ctx):
    """Group -1 handler: consumes text only while /otctest or /realtest waits for it, otherwise lets the normal handlers run."""
    waiting = [m for m in ("otc", "real") if S[m]["step"] is not None]
    if not waiting or not D["allowed"](update):
        return
    m = LAST["m"] if LAST["m"] in waiting else waiting[0]
    st = S[m]
    msg = update.effective_message
    txt = msg.text or ""
    if st["step"] == "paste":
        st["text"] += txt + "\n"
        await msg.reply_text("%s: পেয়েছি (এ পর্যন্ত %d লাইন)। আরও থাকলে পাঠাও, শেষ হলে ✅ চাপো।" % (NAME[m], st["text"].count("\n")), reply_markup=_kb_paste(m))
        raise ApplicationHandlerStop
    if st["step"] == "date":
        today = dt.datetime.now(BDT).date()
        ds = parse_dates(txt, today)
        if not ds:
            await msg.reply_text("তারিখ বুঝিনি। এভাবে লেখো: 17 / 17,18,19 / 17-19 / 17/11 / 2026-11-17 / গতকাল", reply_markup=_kb_date(m))
            raise ApplicationHandlerStop
        await _start(ctx, msg.chat_id, ds, m)
        raise ApplicationHandlerStop


async def _start(ctx, chat_id, dates, m):
    st = S[m]
    today = dt.datetime.now(BDT).date()
    if any(d > today for d in dates):
        return await D["send"](chat_id, "ভবিষ্যতের তারিখ দিয়েছ। আজ বা আগের তারিখ দাও।")
    if len(dates) > MAX_DAYS:
        return await D["send"](chat_id, "একসাথে সর্বোচ্চ %d দিন। কম দিন দাও।" % MAX_DAYS)
    if st["running"]:
        return
    st["step"] = None
    st["running"] = True
    st["stop"] = False
    ctx.application.bot_data["otctest_job_" + m] = asyncio.create_task(_job(chat_id, list(dates), m))


async def _job(chat_id, dates, m):
    send = D["send"]
    st = S[m]
    hdr, rules = st["hdr"], st["rules"]
    dest = _dest(chat_id, m)
    delay = 3.2 if str(dest) != str(chat_id) else 1.1
    kb_stop = Markup([[Btn("⏹ বন্ধ করো", callback_data="%s:stop:0" % PFX[m])]])
    try:
        prog = await send(chat_id, "⏳ %s টেস্ট: %s-এর ক্যান্ডেল পড়ছি ও রিপ্লে করছি…" % (NAME[m], ", ".join(d.strftime("%d %b") for d in dates)), reply_markup=kb_stop)
        async with D["lock"]:
            sigs, info = await asyncio.to_thread(replay, rules, hdr, dates)
        if st["mode"] == "all" and sigs:
            if str(dest) != str(chat_id):
                await send(chat_id, "📨 %d টা সিগন্যালের মেসেজ চ্যানেলে যাচ্ছে (প্রায় %d মিনিট লাগবে)।" % (len(sigs), max(1, round(len(sigs) * 2 * delay / 60))))
            tally = {}
            last_day = None
            for s in sigs:
                if st["stop"]:
                    await send(chat_id, "⏹ বন্ধ করা হয়েছে।")
                    break
                if s["date"] != last_day:
                    last_day = s["date"]
                    await send(dest, "━━━━━━━━━━━━\n📅 %s-এর রিপ্লে শুরু (%s)" % (s["date"].strftime("%d %b %Y"), NAME[m]))
                    await asyncio.sleep(delay)
                await send(dest, fmt_signal(s, hdr))
                await asyncio.sleep(delay)
                t = tally.setdefault(s["date"], [0, 0])
                t[0 if s["res"] == 1 else 1] += 1
                await send(dest, fmt_result(s, t))
                await asyncio.sleep(delay)
        for t in summary_texts(sigs, info, hdr, dates, rules):
            await send(dest, t, parse_mode="HTML")
            if str(dest) != str(chat_id):
                await send(chat_id, t, parse_mode="HTML")
        if sigs:
            bio = io.BytesIO(make_csv(sigs))
            bio.name = "%stest_%s.csv" % (PFX[m] if m == "otc" else "real", dt.datetime.now(BDT).strftime("%Y%m%d_%H%M"))
            try:
                await D["bot"]().send_document(chat_id, bio, caption="প্রতিটা সিগন্যালের তারিখ, সময় ও রেজাল্ট")
            except Exception:  # noqa
                log.exception("otctest csv failed")
        if prog:
            try:
                await prog.delete()
            except Exception:  # noqa
                pass
    except Exception as e:  # noqa
        log.exception("otctest failed")
        await send(chat_id, "❌ %s টেস্ট হয়নি: %s" % (NAME[m], e))
    finally:
        st["running"] = False


async def on_cb(update, ctx):
    q = update.callback_query
    if not D["allowed"](update):
        return await q.answer("অনুমতি নেই", show_alert=True)
    pfx, key, val = q.data.split(":")[:3]
    m = "real" if pfx == "rt" else "otc"
    st = S[m]
    if key == "stop":
        st["stop"] = True
        return await q.answer("বন্ধ করছি…")
    if key == "cancel":
        _reset(m)
        await q.answer("বাতিল")
        try:
            await q.edit_message_text("%s টেস্ট বাতিল।" % NAME[m])
        except BadRequest:
            pass
        return
    if key == "done":
        if st["step"] != "paste":
            return await q.answer("আগে %s দাও" % CMD[m], show_alert=True)
        hdr, rules, err = parse(st["text"])
        if not err:
            other = [r for r in rules if _mk(r["pair"]) != m]
            if other:
                if len(other) == len(rules):
                    err = "এটা %s তালিকা। %s দাও।" % ("রিয়েল" if m == "otc" else "OTC", CMD["real" if m == "otc" else "otc"])
                else:
                    err = "তালিকায় রিয়েল ও OTC মিশে আছে। আলাদা আলাদা তালিকা দিয়ে %s আর %s চালাও।" % (CMD["otc"], CMD["real"])
        if err:
            return await q.answer(err[:190], show_alert=True)
        st["hdr"], st["rules"], st["step"] = hdr, rules, "date"
        LAST["m"] = m
        await q.answer()
        await q.message.reply_text("📋 %s তালিকা পড়লাম:\n%s\n\nকোন তারিখে টেস্ট করবে? লেখো: 17 / 17,18,19 / 17-19 / 17/11 / গতকাল\n(তালিকা বানানোর পরের দিনের তারিখ দাও, নইলে ফলাফল চিট হবে)" % (
            NAME[m], summary_line(hdr, rules)), reply_markup=_kb_date(m))
        return
    if key == "mode":
        st["mode"] = "sum" if st["mode"] == "all" else "all"
        await q.answer()
        try:
            await q.edit_message_reply_markup(reply_markup=_kb_date(m))
        except BadRequest:
            pass
        return
    if key == "day":
        if st["step"] != "date":
            return await q.answer("আগে %s দিয়ে তালিকা পাঠাও" % CMD[m], show_alert=True)
        today = dt.datetime.now(BDT).date()
        await q.answer("শুরু হচ্ছে…")
        await _start(ctx, q.message.chat_id, [today - dt.timedelta(days=1)] if val == "y" else [today], m)


def register(app, send, allowed, deny, lock, bot_getter):
    """Called once from bot.py main(): own commands (/otctest, /realtest), callbacks and a text handler (group -1) — nothing existing is changed."""
    D.update(send=send, allowed=allowed, deny=deny, lock=lock, bot=bot_getter)
    app.add_handler(CommandHandler("otctest", cmd_otctest))
    app.add_handler(CommandHandler("realtest", cmd_realtest))
    app.add_handler(CallbackQueryHandler(on_cb, pattern=r"^(ot|rt):"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text), group=-1)


COMMAND = BotCommand("otctest", "OTC টেস্ট: /engine10 OTC তালিকা পেস্ট করে কোনো দিনের রিপ্লে")
COMMAND_REAL = BotCommand("realtest", "রিয়েল টেস্ট: /engine10 রিয়েল তালিকা পেস্ট করে কোনো দিনের রিপ্লে")

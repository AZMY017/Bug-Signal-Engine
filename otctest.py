"""/otctest — replay a saved /engine10 radar list on the stored candles of chosen day(s), like a live run (no live data needed).

The list is the TEXT of an /engine10 result that the user pastes into the chat, so the list is frozen (it can not be changed by
the candles of the test day). The bot reads tf, MTG, N back-to-back losses ("exactly N" / "N or more") and every pair's
green / red minutes from that text, asks for the day(s), reads the M1 candles of that day from Neon (read-only, via source.load_oc),
and replays every slot with the SAME code as the back-test (engine4.outcomes): each hour the slot's minute is traded, N losses in a
row (the streak restarts every day) -> the next hour's trade is a SIGNAL; its result is a direct win / MTG win / loss (doji = win).
Messages: signal, then result (in order of time), then a summary + CSV. Optional separate chat: env OTC_TEST_CHAT_ID.

Nothing of the other engines is touched; this file only registers its own command, callbacks and a text handler.
"""
import asyncio
import csv
import datetime as dt
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
TEST_CHAT = os.environ.get("OTC_TEST_CHAT_ID", "").strip()          # optional channel / group for the replay messages
UP, DOWN = "UP (Call) 📈", "DOWN (Put) 📉"
MAX_DAYS = 7                                                          # most days in one test

OT = dict(step=None, text="", rules=None, hdr=None, mode="all", stop=False, running=False)   # step: None | "paste" | "date"
D = {}          # injected from bot.py: send, allowed, deny, lock


def _reset():
    OT.update(step=None, text="", rules=None, hdr=None, stop=False)


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


# ---------------------------------------------------------------- 3) replay (same code as the back-test)
def replay(rules, hdr, dates):
    """-> (signals list, info dict). signal = dict(ts, date, pair, side, minute, hour, mtg, res 1|-1, lvl, doji, streak, loss_hours).
    info: no_data[(pair, date)], partial[(pair, date, last 'HH:MM')], skipped_missing (int)."""
    tf, n, atleast = hdr["tf"], hdr["n"], hdr["atleast"]
    info = dict(no_data=[], partial=[], missing=0)
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
    return ("🧪 OTC টেস্ট (রিপ্লে) • %s\n💱 %s   ⏰ %02d:%02d (UTC+6)\n%s\n⏱ টাইম ফ্রেম: %d মিনিট\n🔁 MTG: %s\nℹ️ এই স্লটে পরপর %d লস: %s" % (
        s["date"].strftime("%d %b"), pairs.pretty(s["pair"]), s["hour"], s["minute"], side, hdr["tf"], _mtg_txt(s["mtg"]), s["streak"], done))


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


def summary_texts(sigs, info, hdr, dates, rules, partial_note=True):
    n, w, l, lv = stats(sigs)
    ds = ", ".join(d.strftime("%d %b") for d in dates)
    L = ["📊 OTC টেস্ট রিপোর্ট • %s" % ds, summary_line(hdr, rules), ""]
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
        L.append("")
        L.append("পেয়ার ধরে:")
        for p in sorted({s["pair"] for s in sigs}):
            ss = [s for s in sigs if s["pair"] == p]
            a, b, c_, _ = stats(ss)
            L.append("  %s: %d • ✅ %d • ❌ %d" % (pairs.pretty(p), a, b, c_))
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
    txt = "\n".join(L)
    out, cur = [], ""
    for line in txt.split("\n"):
        if len(cur) + len(line) + 1 > 3900:
            out.append(cur)
            cur = ""
        cur += line + "\n"
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


# ---------------------------------------------------------------- 5) Telegram glue
def _kb_paste():
    return Markup([[Btn("✅ পাঠানো শেষ", callback_data="ot:done:0"), Btn("❌ বাতিল", callback_data="ot:cancel:0")]])


def _kb_date():
    return Markup([[Btn("📨 মেসেজ: সব (লাইভের মতো)" if OT["mode"] == "all" else "📊 মেসেজ: শুধু সারাংশ", callback_data="ot:mode:0")],
                   [Btn("গতকাল", callback_data="ot:day:y"), Btn("আজ", callback_data="ot:day:t")],
                   [Btn("❌ বাতিল", callback_data="ot:cancel:0")]])


async def cmd_otctest(update, ctx):
    if not D["allowed"](update):
        return await D["deny"](update)
    if OT["running"]:
        return await update.effective_message.reply_text("একটা টেস্ট চলছে। বন্ধ করতে নিচের ⏹ বাটন চাপো, বা শেষ হওয়া পর্যন্ত অপেক্ষা করো।")
    if not source.otc_enabled() and not source.enabled():
        return await update.effective_message.reply_text("OTC_DATABASE_URL_* / DATABASE_URL_* সেট করা নেই।")
    _reset()
    OT["step"] = "paste"
    await update.effective_message.reply_text(
        "🧪 OTC টেস্ট (রিপ্লে)\n\n/engine10-এর রেজাল্ট টেক্সট এখানে পাঠাও (কপি করে পেস্ট)। একাধিক মেসেজ হলে সবগুলো পাঠাও, তারপর ✅ চাপো।\n"
        "প্রথম মেসেজে হেডার লাইন (টাইম ফ্রেম, MTG, পরপর লস) থাকতে হবে।", reply_markup=_kb_paste())


async def on_text(update, ctx):
    """Group -1 handler: consumes text only while /otctest waits for it, otherwise lets the normal handlers run."""
    if OT["step"] is None or not D["allowed"](update):
        return
    msg = update.effective_message
    txt = msg.text or ""
    if OT["step"] == "paste":
        OT["text"] += txt + "\n"
        await msg.reply_text("পেয়েছি (এ পর্যন্ত %d লাইন)। আরও থাকলে পাঠাও, শেষ হলে ✅ চাপো।" % OT["text"].count("\n"), reply_markup=_kb_paste())
        raise ApplicationHandlerStop
    if OT["step"] == "date":
        today = dt.datetime.now(BDT).date()
        ds = parse_dates(txt, today)
        if not ds:
            await msg.reply_text("তারিখ বুঝিনি। এভাবে লেখো: 17 / 17,18,19 / 17-19 / 17/11 / 2026-11-17 / গতকাল", reply_markup=_kb_date())
            raise ApplicationHandlerStop
        await _start(ctx, msg.chat_id, ds)
        raise ApplicationHandlerStop


async def _start(ctx, chat_id, dates):
    today = dt.datetime.now(BDT).date()
    if any(d > today for d in dates):
        return await D["send"](chat_id, "ভবিষ্যতের তারিখ দিয়েছ। আজ বা আগের তারিখ দাও।")
    if len(dates) > MAX_DAYS:
        return await D["send"](chat_id, "একসাথে সর্বোচ্চ %d দিন। কম দিন দাও।" % MAX_DAYS)
    if OT["running"]:
        return
    OT["step"] = None
    OT["running"] = True
    OT["stop"] = False
    ctx.application.bot_data["otctest_job"] = asyncio.create_task(_job(chat_id, list(dates)))


async def _job(chat_id, dates):
    send = D["send"]
    hdr, rules = OT["hdr"], OT["rules"]
    dest = TEST_CHAT or chat_id
    try:
        dest = int(dest)
    except (TypeError, ValueError):
        pass
    delay = 3.2 if str(dest) != str(chat_id) else 1.1
    try:
        prog = await send(chat_id, "⏳ OTC টেস্ট: %s-এর ক্যান্ডেল পড়ছি ও রিপ্লে করছি…" % ", ".join(d.strftime("%d %b") for d in dates),
                          reply_markup=Markup([[Btn("⏹ বন্ধ করো", callback_data="ot:stop:0")]]))
        async with D["lock"]:
            sigs, info = await asyncio.to_thread(replay, rules, hdr, dates)
        if OT["mode"] == "all" and sigs:
            if str(dest) != str(chat_id):
                await send(chat_id, "📨 %d টা সিগন্যালের মেসেজ চ্যানেলে যাচ্ছে (প্রায় %d মিনিট লাগবে)।" % (len(sigs), max(1, round(len(sigs) * 2 * delay / 60))))
            tally = {}
            last_day = None
            for s in sigs:
                if OT["stop"]:
                    await send(chat_id, "⏹ বন্ধ করা হয়েছে।")
                    break
                if s["date"] != last_day:
                    last_day = s["date"]
                    await send(dest, "━━━━━━━━━━━━\n📅 %s-এর রিপ্লে শুরু" % s["date"].strftime("%d %b %Y"))
                    await asyncio.sleep(delay)
                await send(dest, fmt_signal(s, hdr))
                await asyncio.sleep(delay)
                t = tally.setdefault(s["date"], [0, 0])
                t[0 if s["res"] == 1 else 1] += 1
                await send(dest, fmt_result(s, t))
                await asyncio.sleep(delay)
        for t in summary_texts(sigs, info, hdr, dates, rules):
            await send(dest, t)
            if str(dest) != str(chat_id):
                await send(chat_id, t)
        if sigs:
            bio = io.BytesIO(make_csv(sigs))
            bio.name = "otctest_%s.csv" % dt.datetime.now(BDT).strftime("%Y%m%d_%H%M")
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
        await send(chat_id, "❌ OTC টেস্ট হয়নি: %s" % e)
    finally:
        OT["running"] = False


async def on_cb(update, ctx):
    q = update.callback_query
    if not D["allowed"](update):
        return await q.answer("অনুমতি নেই", show_alert=True)
    key = q.data.split(":")[1]
    val = q.data.split(":")[2]
    if key == "stop":
        OT["stop"] = True
        return await q.answer("বন্ধ করছি…")
    if key == "cancel":
        _reset()
        await q.answer("বাতিল")
        try:
            await q.edit_message_text("OTC টেস্ট বাতিল।")
        except BadRequest:
            pass
        return
    if key == "done":
        if OT["step"] != "paste":
            return await q.answer("আগে /otctest দাও", show_alert=True)
        hdr, rules, err = parse(OT["text"])
        if err:
            return await q.answer(err[:190], show_alert=True)
        OT["hdr"], OT["rules"], OT["step"] = hdr, rules, "date"
        await q.answer()
        warn = ""
        otc = [r for r in rules if pairs.is_otc(r["pair"])]
        real = len(rules) - len(otc)
        if real and otc:
            warn = "\n⚠️ তালিকায় রিয়েল ও OTC দুটোই আছে, দুটোই চলবে।"
        await q.message.reply_text("📋 তালিকা পড়লাম:\n%s%s\n\nকোন তারিখে টেস্ট করবে? লেখো: 17 / 17,18,19 / 17-19 / 17/11 / গতকাল\n(তালিকা বানানোর পরের দিনের তারিখ দাও, নইলে ফলাফল চিট হবে)" % (
            summary_line(hdr, rules), warn), reply_markup=_kb_date())
        return
    if key == "mode":
        OT["mode"] = "sum" if OT["mode"] == "all" else "all"
        await q.answer()
        try:
            await q.edit_message_reply_markup(reply_markup=_kb_date())
        except BadRequest:
            pass
        return
    if key == "day":
        if OT["step"] != "date":
            return await q.answer("আগে /otctest দিয়ে তালিকা পাঠাও", show_alert=True)
        today = dt.datetime.now(BDT).date()
        await q.answer("শুরু হচ্ছে…")
        await _start(ctx, q.message.chat_id, [today - dt.timedelta(days=1)] if val == "y" else [today])


def register(app, send, allowed, deny, lock, bot_getter):
    """Called once from bot.py main(): own command, callbacks and a text handler (group -1) — nothing existing is changed."""
    D.update(send=send, allowed=allowed, deny=deny, lock=lock, bot=bot_getter)
    app.add_handler(CommandHandler("otctest", cmd_otctest))
    app.add_handler(CallbackQueryHandler(on_cb, pattern=r"^ot:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text), group=-1)


COMMAND = BotCommand("otctest", "OTC টেস্ট: /engine10 তালিকা পেস্ট করে কোনো দিনের রিপ্লে")

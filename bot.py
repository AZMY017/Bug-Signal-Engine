"""Bug Signal Engine — separate Telegram bot. Reads candles from the live bot's Neon databases (read-only), finds
time-slot "bug" patterns, saves them in its own results database and (optionally) sends live alerts."""
import asyncio
import csv
import datetime as dt
import io
import json
import logging
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

from telegram import BotCommand, InlineKeyboardButton as Btn, InlineKeyboardMarkup as Markup, Update
from telegram.error import BadRequest, RetryAfter
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

import engine
import engine2
import engine3
import live
import oanda
import pairs
import resultsdb as rdb
import source

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bugbot")

BOT_TOKEN = os.environ["BOT_TOKEN"]
ALLOWED = {int(x) for x in os.environ.get("ALLOWED_USER_IDS", "").replace(" ", "").split(",") if x}
LIVE_CHAT_ID = os.environ.get("LIVE_CHAT_ID", "").strip()        # optional: channel/group id for alerts (else the chat where you press "চালু")


def _f(name, default):
    return float(os.environ.get(name, default))


PARAMS = dict(min_wr=_f("MIN_WR", "0.80"), min_edge=_f("MIN_EDGE", "0.03"), z=_f("SCAN_Z", "3.5"),
              min_n=int(_f("MIN_N", "100")), min_n_cond=int(_f("MIN_N_COND", "30")),
              min_test_n=int(_f("MIN_TEST_N", "25")), min_test_n_cond=int(_f("MIN_TEST_N_COND", "8")),
              split=_f("SPLIT", "0.6667"))
SCAN_DAYS = int(_f("SCAN_DAYS", "400"))
MAX_KEEP = int(_f("MAX_KEEP", "3000"))
AUTO_SCAN_HOURS = _f("AUTO_SCAN_HOURS", "24")
WINDOWS = engine.WINDOWS_DEFAULT
if os.environ.get("WINDOWS", "").strip():
    try:
        WINDOWS = [tuple(int(x) for x in w.split("-")) for w in os.environ["WINDOWS"].split(",")]
        assert all(0 <= a < b <= 24 for a, b in WINDOWS)
    except Exception:  # noqa
        log.error("WINDOWS format wrong (use 0-24,12-24), using default")
        WINDOWS = engine.WINDOWS_DEFAULT

APP = None
SCAN_LOCK = asyncio.Lock()
SCAN = {"text": "এখনো স্ক্যান চলেনি", "at": 0.0}


def allowed(update):
    u = update.effective_user
    return bool(u and u.id in ALLOWED)


async def deny(update):
    uid = update.effective_user.id if update.effective_user else "?"
    await update.effective_message.reply_text("এই বট ব্যক্তিগত। তোমার Telegram ID: %s (ALLOWED_USER_IDS-এ বসাও)" % uid)


async def send(chat, text, **kw):
    for _ in range(3):
        try:
            return await APP.bot.send_message(chat, text, **kw)
        except RetryAfter as e:
            await asyncio.sleep(float(e.retry_after) + 1)
        except Exception:  # noqa
            log.exception("send failed")
            return None


def chunks(text, n=3900):
    out, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > n:
            out.append(cur)
            cur = ""
        cur += line + "\n"
    if cur.strip():
        out.append(cur)
    return out


# ---------------------------------------------------------------- formatting
def pct(w, n):
    return "%d%%" % round(100.0 * w / n) if n else "—"


def fmt_rule(i, r):
    cond = "সরাসরি" if r["nloss"] == 0 else "%d লসের পর" % r["nloss"]
    mtg = "No-MTG" if r["mtg"] == 0 else "MTG%d" % r["mtg"]
    flag = "✅" if r["robust"] else "⚠️"
    return ("%s%d. %s | :%02d %s | %02d–%02d | %s | %s\n"
            "    %s (%d/%d) • টেস্ট %s (%d) • সর্বোচ্চ ব্যাক-টু-ব্যাক লস %d" % (
                flag, i, pairs.pretty(r["pair"]), r["minute"], r["side"], r["hs"], r["he"], mtg, cond,
                pct(r["w"], r["n"]), r["w"], r["n"], pct(r["wte"], r["nte"]), r["nte"], r["maxl"]))


LEGEND = ("✅ = টেস্ট পিরিয়ডেও টিকেছে • ⚠️ = শুধু আগের ডেটায় ভালো\n"
          "সময় UTC+6 • :৪৫ মানে প্রতি ঘণ্টার ৪৫ মিনিটের ক্যান্ডেল • 12–24 = দুপুর ১২টা থেকে রাত ১২টা")


def head_cmds():
    return ("/scan — এখনই স্ক্যান\n/bugs — পাওয়া বাগের তালিকা\n/csv — তালিকা CSV ফাইলে\n"
            "/check USDJPY 30 UP 12-24 — একটা নির্দিষ্ট স্লটের হিসাব\n/engine2 — নতুন ইঞ্জিন ২ (Back Signal Discovery)\n/engine3 — ইঞ্জিন ৩ (মিনিট গ্রিড: G/R টেবিল + উইন রেশিও, PDF/CSV)\n/live — লাইভ অ্যালার্ট চালু/বন্ধ\n/status — অবস্থা\n/data — ডেটা কভারেজ\n/id — আমার Telegram ID")


# ---------------------------------------------------------------- scan
async def do_scan(progress, min_wr=None, only=None):
    """Returns (run_id, summary_text, new_rules_text). Raises RuntimeError with a Bengali message on setup problems."""
    if not source.enabled():
        raise RuntimeError("DATABASE_URL_1/2/3 সেট করা নেই (লাইভ বটের ক্যান্ডেল ডেটাবেজ)।")
    if not rdb.enabled():
        raise RuntimeError("RESULTS_DATABASE_URL সেট করা নেই (ইঞ্জিনের নিজের আলাদা Neon ডেটাবেজ)।")
    async with SCAN_LOCK:
        p = dict(PARAMS)
        if min_wr:
            p["min_wr"] = min_wr
        t0 = time.time()
        prev = await asyncio.to_thread(rdb.latest_run)
        old_keys = await asyncio.to_thread(rdb.rule_keys, prev[0]) if prev else set()
        run_id = await asyncio.to_thread(rdb.new_run, json.dumps(dict(p, windows=WINDOWS)))
        plist = only or pairs.PAIRS
        rows_all, used, missing = [], [], []
        near_all = []
        p["near_out"] = near_all
        now = int(time.time())
        try:
            for i, pair in enumerate(plist):
                await progress("⏳ স্ক্যান চলছে: %d/%d • %s" % (i + 1, len(plist), pairs.pretty(pair)))
                ts, d = await asyncio.to_thread(source.load_dir, pair, now - SCAN_DAYS * 86400, now + 86400)
                if len(ts) < 5000:
                    missing.append(pairs.pretty(pair))
                    continue
                rows = await asyncio.to_thread(engine.scan_pair, pair, ts, d, WINDOWS, p)
                rows_all += rows
                used.append((pairs.pretty(pair), len(ts), int(ts[0]), int(ts[-1])))
            near_all.sort(key=lambda r: r["score"], reverse=True)
            rows_all.sort(key=lambda r: (r["robust"], r["score"]), reverse=True)
            rows_all = rows_all[:MAX_KEEP]
            await progress("💾 রেজাল্ট সেভ হচ্ছে…")
            for k in range(0, len(rows_all), 500):
                await asyncio.to_thread(rdb.add_rows, run_id, rows_all[k:k + 500])
            n_rob = sum(1 for r in rows_all if r["robust"])
            note = json.dumps(dict(pairs=len(used), missing=missing, rows=len(rows_all), robust=n_rob,
                                   took=int(time.time() - t0), min_wr=p["min_wr"]))
            p.pop("near_out", None)
            await asyncio.to_thread(rdb.finish_run, run_id, "done", note)
        except Exception as e:
            await asyncio.to_thread(rdb.finish_run, run_id, "failed", str(e)[:200])
            raise
        live.invalidate()
    days = ""
    if used:
        days = "%d দিনের মতো ডেটা" % ((max(u[3] for u in used) - min(u[2] for u in used)) // 86400)
    rob = engine.dedupe([r for r in rows_all if r["robust"]])
    txt = "🐞 স্ক্যান শেষ (%d সেকেন্ড)\n%d পেয়ার স্ক্যান হলো • %s\n" % (int(time.time() - t0), len(used), days)
    if missing:
        txt += "ডেটা নেই/কম: %s\n" % ", ".join(missing)
    txt += "টেস্ট-পাস বাগ: %d টা (একই বাগের ভ্যারিয়েন্ট এক ধরে) • মোট প্রার্থী (সব ভ্যারিয়েন্ট): %d টা\n" % (len(rob), len(rows_all))
    if rob:
        txt += "\nসেরা ৮টা:\n" + "\n".join(fmt_rule(i + 1, r) for i, r in enumerate(rob[:8])) + "\n\n" + LEGEND
    else:
        txt += "\nএই শর্তে কোনো বাগ টেস্টে টেকেনি।"
        if near_all:
            txt += (" সবচেয়ে কাছাকাছি ৬টা (পাস করেনি) দেখো:\n" +
                    "\n".join(fmt_rule(i + 1, dict(r, robust=False)) for i, r in enumerate(engine.dedupe(near_all)[:6])) +
                    "\n\n'কয়েন-ফ্লিপ' = শুধু ভাগ্যে যা আসত (MTG ছাড়া ৫০%%, MTG১ ৭৫%%, MTG২ ৮৭.৫%%)। কাছাকাছি তালিকার WR এর কাছাকাছি হলে ওটা বাগ না, শুধু ভাগ্য।")
    new_txt = ""
    if prev:
        fresh = [r for r in rob if "|".join(str(r[k]) for k in ("pair", "minute", "side", "mtg", "hs", "he", "nloss")) not in old_keys]
        if fresh:
            new_txt = "🆕 আগের স্ক্যানের পর নতুন বাগ (%d টা):\n" % len(fresh) + "\n".join(fmt_rule(i + 1, r) for i, r in enumerate(fresh[:5]))
    SCAN.update(text=txt.split("\n")[0], at=time.time())
    return run_id, txt, new_txt


def parse_scan_args(args):
    min_wr, only = None, []
    for a in args:
        for tok in a.replace(",", " ").split():
            try:
                v = float(tok)
                min_wr = v / 100.0 if v >= 1 else v
            except ValueError:
                p = pairs.norm(tok)
                if p:
                    only.append(p)
    return min_wr, only or None


async def cmd_scan(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return await deny(update)
    msg = update.effective_message
    if SCAN_LOCK.locked():
        return await msg.reply_text("স্ক্যান আগে থেকেই চলছে, শেষ হলে জানাবো।")
    min_wr, only = parse_scan_args(ctx.args or [])
    prog = await msg.reply_text("⏳ স্ক্যান শুরু…")

    async def progress(t):
        try:
            await prog.edit_text(t)
        except BadRequest:
            pass

    async def job():
        try:
            _, txt, new_txt = await do_scan(progress, min_wr, only)
        except Exception as e:  # noqa
            log.exception("scan failed")
            return await progress("❌ স্ক্যান হয়নি: %s" % e)
        try:
            await prog.delete()
        except Exception:  # noqa
            pass
        for part in chunks(txt):
            await send(msg.chat_id, part)
        if new_txt:
            await send(msg.chat_id, new_txt)

    ctx.application.bot_data["scan_job"] = asyncio.create_task(job())


# ---------------------------------------------------------------- /bugs, /csv
PAGE = 8


async def bugs_page(robust, page):
    run = await asyncio.to_thread(rdb.latest_run)
    if not run:
        return "এখনো কোনো স্ক্যান হয়নি। /scan চাপো।", None
    if robust:
        allr = engine.dedupe(await asyncio.to_thread(rdb.get_bugs, run[0], True, 600, 0))
        total = len(allr)
        rows = allr[page * PAGE:(page + 1) * PAGE]
    else:
        total = await asyncio.to_thread(rdb.count_bugs, run[0], False)
        rows = await asyncio.to_thread(rdb.get_bugs, run[0], False, PAGE, page * PAGE)
    when = dt.datetime.fromtimestamp(run[2], pairs.BDT).strftime("%d/%m %H:%M")
    if not rows:
        return "এই তালিকা ফাঁকা। (স্ক্যান: %s)" % when, None
    txt = "%s তালিকা • মোট %d • স্ক্যান %s\n\n" % ("টেস্ট-পাস (ভ্যারিয়েন্ট মেলানো)" if robust else "সব প্রার্থী (সব ভ্যারিয়েন্ট)", total, when)
    txt += "\n".join(fmt_rule(page * PAGE + i + 1, r) for i, r in enumerate(rows)) + "\n\n" + LEGEND
    nav = []
    if page > 0:
        nav.append(Btn("◀", callback_data="bg:%d:%d" % (robust, page - 1)))
    if (page + 1) * PAGE < total:
        nav.append(Btn("▶", callback_data="bg:%d:%d" % (robust, page + 1)))
    rows_kb = [nav] if nav else []
    rows_kb.append([Btn("সব প্রার্থী দেখো" if robust else "শুধু টেস্ট-পাস দেখো", callback_data="bg:%d:0" % (0 if robust else 1))])
    return txt, Markup(rows_kb)


async def cmd_bugs(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return await deny(update)
    try:
        txt, kb = await bugs_page(1, 0)
    except Exception as e:  # noqa
        return await update.effective_message.reply_text("ডেটাবেজ সমস্যা: %s" % e)
    await update.effective_message.reply_text(txt[:4000], reply_markup=kb)


async def on_bugs_cb(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not allowed(update):
        return await q.answer("অনুমতি নেই", show_alert=True)
    await q.answer()
    _, r, p = q.data.split(":")
    txt, kb = await bugs_page(int(r), int(p))
    try:
        await q.edit_message_text(txt[:4000], reply_markup=kb)
    except BadRequest:
        pass


async def cmd_csv(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return await deny(update)
    run = await asyncio.to_thread(rdb.latest_run)
    if not run:
        return await update.effective_message.reply_text("এখনো কোনো স্ক্যান হয়নি।")
    rows = await asyncio.to_thread(rdb.get_bugs, run[0], False, 5000, 0)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["pair", "minute", "side", "window_bdt", "mtg", "after_losses", "wins", "total", "winrate_%",
                "test_wins", "test_total", "test_winrate_%", "max_back2back_loss", "coin_flip_%", "robust"])
    for r in rows:
        w.writerow([pairs.pretty(r["pair"]), r["minute"], r["side"], "%02d-%02d" % (r["hs"], r["he"]), r["mtg"],
                    r["nloss"], r["w"], r["n"], round(100.0 * r["w"] / r["n"], 1), r["wte"], r["nte"],
                    round(100.0 * r["wte"] / r["nte"], 1) if r["nte"] else "", r["maxl"], round(100 * r["base"], 1),
                    "yes" if r["robust"] else "no"])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig"))
    data.name = "bug_signals_%s.csv" % dt.datetime.now(pairs.BDT).strftime("%Y%m%d_%H%M")
    await update.effective_message.reply_document(data, caption="%d টা প্রার্থী (টেস্ট-পাস আগে)" % len(rows))


# ---------------------------------------------------------------- /live
def live_text(cf, rules):
    st = "🟢 চালু" if cf["on"] else "🔴 বন্ধ"
    t = "লাইভ অ্যালার্ট: %s\nওয়াচলিস্ট: সেরা %d টা টেস্ট-পাস বাগ (প্রতি পেয়ার/মিনিট/দিক/উইন্ডোতে সেরা ভ্যারিয়েন্ট)\n" % (st, cf["top"])
    t += "অ্যালার্ট যায় স্লটের ~%d সেকেন্ড আগে। শুধু যে পেয়ারের স্লট আসছে, তার আজকের ক্যান্ডেল OANDA থেকে নেয় (Neon-এ হাত দেয় না)।\n" % live.LEAD
    if not oanda.enabled():
        t += "❌ OANDA_KEY সেট করা নেই, অ্যালার্ট কাজ করবে না।\n"
    if rules:
        t += "\nএখন যা দেখছে:\n" + "\n".join(
            "• %s :%02d %s %02d–%02d %s%s" % (pairs.pretty(r["pair"]), r["minute"], r["side"], r["hs"], r["he"],
                                              "No-MTG" if r["mtg"] == 0 else "MTG%d" % r["mtg"],
                                              "" if r["nloss"] == 0 else " (%d লসের পর)" % r["nloss"]) for r in rules[:20])
    else:
        t += "\nওয়াচলিস্ট ফাঁকা — আগে /scan চালাও।"
    return t


def live_kb(cf):
    return Markup([[Btn("🔴 বন্ধ করো" if cf["on"] else "🟢 চালু করো", callback_data="lv:" + ("off" if cf["on"] else "on"))],
                   [Btn(("✔ " if cf["top"] == n else "") + "Top %d" % n, callback_data="lv:top:%d" % n) for n in (5, 10, 20)]])


async def cmd_live(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return await deny(update)
    try:
        cf = await live.cfg(force=True)
        rules = await live.watchlist(force=True)
    except Exception as e:  # noqa
        return await update.effective_message.reply_text("ডেটাবেজ সমস্যা: %s" % e)
    await update.effective_message.reply_text(live_text(cf, rules)[:4000], reply_markup=live_kb(cf))


async def on_live_cb(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not allowed(update):
        return await q.answer("অনুমতি নেই", show_alert=True)
    await q.answer()
    parts = q.data.split(":")
    if parts[1] == "on":
        chat = int(LIVE_CHAT_ID) if LIVE_CHAT_ID.lstrip("-").isdigit() else q.message.chat_id
        await live.set_cfg(on=True, chat=chat)
    elif parts[1] == "off":
        await live.set_cfg(on=False)
    elif parts[1] == "top":
        await live.set_cfg(top=int(parts[2]))
        live.invalidate()
    cf = await live.cfg(force=True)
    rules = await live.watchlist(force=True)
    try:
        await q.edit_message_text(live_text(cf, rules)[:4000], reply_markup=live_kb(cf))
    except BadRequest:
        pass


# ---------------------------------------------------------------- /status /data /id
def ago(t):
    if not t:
        return "—"
    s = int(time.time() - t)
    return "%d সেকেন্ড আগে" % s if s < 120 else "%d মিনিট আগে" % (s // 60) if s < 7200 else "%d ঘণ্টা আগে" % (s // 3600)


async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return await deny(update)
    L = []
    L.append("ক্যান্ডেল ডেটাবেজ (পড়ার জন্য): %s" % ("%d টা" % len(source.URLS) if source.enabled() else "❌ সেট করা নেই"))
    L.append("রেজাল্ট ডেটাবেজ: %s" % ("✅" if rdb.enabled() else "❌ সেট করা নেই"))
    L.append("OANDA (লাইভ অ্যালার্টের ক্যান্ডেল): %s • রিকোয়েস্ট এই সেশনে %d টা%s" % (
        "✅" if oanda.enabled() else "❌ OANDA_KEY নেই", live.DIAG["requests"],
        (" • সমস্যা: " + oanda.STATE["err"]) if oanda.STATE["err"] else ""))
    try:
        run = await asyncio.to_thread(rdb.latest_run)
        if run:
            note = json.loads(run[3] or "{}")
            L.append("শেষ স্ক্যান: %s • %d পেয়ার • টেস্ট-পাস %s টা" % (ago(run[2]), note.get("pairs", 0), note.get("robust", "?")))
        else:
            L.append("শেষ স্ক্যান: এখনো হয়নি")
        cf = await live.cfg(force=True)
        L.append("লাইভ অ্যালার্ট: %s (Top %d)" % ("🟢 চালু" if cf["on"] else "🔴 বন্ধ", cf["top"]))
        w, l, u, o = live.tally(live.day_start(int(time.time())))
        L.append("আজকের অ্যালার্ট: ✅ %d • ❌ %d • ❔ %d • চলমান %d" % (w, l, u, o))
    except Exception as e:  # noqa
        L.append("ডেটাবেজ সমস্যা: %s" % e)
    D = live.DIAG
    L.append("শেষ অ্যালার্ট-চেক: %s" % ago(D["tick_at"]))
    if D["last_alert"]:
        L.append("শেষ অ্যালার্ট: %s %s (%s)" % (D["last_alert"][1], D["last_alert"][2], ago(D["last_alert"][0])))
    if D["skipped"]:
        L.append("শেষ বাদ পড়া: " + D["skipped"])
    if SCAN_LOCK.locked():
        L.append("⏳ স্ক্যান এখন চলছে")
    L.append("শর্ত: WR ≥ %d%% • নমুনা ≥ %d (লসের পরে ≥ %d) • কয়েন-ফ্লিপের চেয়ে অন্তত +%d%% বেশি" % (
        PARAMS["min_wr"] * 100, PARAMS["min_n"], PARAMS["min_n_cond"], PARAMS["min_edge"] * 100))
    await update.effective_message.reply_text("\n".join(L))


async def cmd_data(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return await deny(update)
    if not source.enabled():
        return await update.effective_message.reply_text("DATABASE_URL_* সেট করা নেই।")
    msg = await update.effective_message.reply_text("ডেটা দেখছি…")
    L = []
    for p in pairs.PAIRS:
        try:
            lo, hi = await asyncio.to_thread(source.bounds, p)
        except Exception as e:  # noqa
            L.append("%s: ❌ %s" % (pairs.pretty(p), e))
            continue
        if lo is None:
            L.append("%s: ডেটা নেই" % pairs.pretty(p))
        else:
            L.append("%s: %d দিন • শেষ ক্যান্ডেল %s" % (pairs.pretty(p), (hi - lo) // 86400,
                                                      dt.datetime.fromtimestamp(hi, pairs.BDT).strftime("%d/%m %H:%M")))
    await msg.edit_text("\n".join(L)[:4000])


async def cmd_check(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/check USDJPY 30 UP 12-24 : numbers of ONE slot (your known bug) over all stored days."""
    if not allowed(update):
        return await deny(update)
    msg = update.effective_message
    a = ctx.args or []
    usage = "ব্যবহার: /check USDJPY 30 UP 12-24\n(পেয়ার, ঘণ্টার কত মিনিট, UP বা DOWN, সময়-উইন্ডো UTC+6; উইন্ডো না দিলে 0-24)"
    try:
        pair = pairs.norm(a[0])
        minute = int(a[1])
        side = {"UP": 1, "DOWN": -1}[a[2].upper()]
        hs, he = (int(x) for x in (a[3] if len(a) > 3 else "0-24").split("-"))
        assert pair and 0 <= minute < 60 and 0 <= hs < he <= 24
    except Exception:  # noqa
        return await msg.reply_text(usage)
    if not source.enabled():
        return await msg.reply_text("DATABASE_URL_* সেট করা নেই।")
    wait = await msg.reply_text("⏳ ডেটা পড়ছি…")
    try:
        now = int(time.time())
        ts, d = await asyncio.to_thread(source.load_dir, pair, now - SCAN_DAYS * 86400, now + 86400)
        if len(ts) < 5000:
            return await wait.edit_text("এই পেয়ারের ডেটা কম/নেই।")
        res = await asyncio.to_thread(engine.check_slot, ts, d, side, minute, hs, he)
    except Exception as e:  # noqa
        return await wait.edit_text("❌ %s" % e)
    L = ["%s | :%02d %s | %02d–%02d | %d দিনের মতো ডেটা" % (pairs.pretty(pair), minute, "UP" if side == 1 else "DOWN", hs, he,
                                                      (ts[-1] - ts[0]) // 86400)]
    for r in res:
        L.append("")
        L.append("%s: উইন রেট %s (%d/%d) • কয়েন-ফ্লিপ হতো %d%% • টেস্ট অংশে %s (%d)" % (
            "No-MTG" if r["mtg"] == 0 else "MTG%d" % r["mtg"], pct(r["w"][0], r["n"][0]), r["w"][0], r["n"][0],
            round(100 * r["base"]), pct(r["wte"], r["nte"]), r["nte"]))
        L.append("   সর্বোচ্চ ব্যাক-টু-ব্যাক লস (একদিনে): %d" % r["maxl"])
        L.append("   লসের পর ট্রেড: " + " • ".join("%d লস → %s (%d)" % (N, pct(r["w"][N], r["n"][N]), r["n"][N]) for N in (1, 2, 3)))
    await wait.edit_text("\n".join(L)[:4000])


async def cmd_id(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("তোমার Telegram ID: %d" % update.effective_user.id)


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return await deny(update)
    await update.effective_message.reply_text(
        "🐞 বাগ সিগন্যাল ইঞ্জিন\n\n"
        "এক বছরের মতো M1 ক্যান্ডেল স্ক্যান করে এমন সময়ভিত্তিক প্যাটার্ন খোঁজে, যেখানে উইন রেট বেশি আর ব্যাক-টু-ব্যাক লস কম, "
        "আর আলাদা টেস্ট পিরিয়ডে যাচাই করে।\n\n" + head_cmds())


# ---------------------------------------------------------------- /engine2  (Back Signal Discovery)
E2 = dict(months=12, side="BOTH", mtg="BOTH", acc=0.80, nmin=300)
E2_PAYOUT = _f("E2_PAYOUT", "0.85")      # broker payout, 0.85 = 85%
E2_MULT = _f("E2_MTG_MULT", "2.2")        # stake multiplier of the MTG step (for the EV number)
E2_Z = _f("E2_Z", "0")                    # 0 = automatic (Bonferroni, depends on how many slots are tested)
E2_ICON = {"valid": "✅", "low": "🟡", "overfit": "❌", "no": "▫️"}
E2_LEGEND = ("✅ = প্রথম অংশে খুঁজে পাওয়া, শেষ অংশে (OOS) আর প্রতি মাসে/তিন ভাগে টিকেছে • 🟡 = নমুনা কম, কম ভরসা • "
             "❌ = ওভারফিট (শুধু আগের ডেটায় ভালো)\n"
             "OOS = শেষ ১/৩ অংশ, যেটা বাগ খোঁজার সময় দেখা হয়নি। 'রিকভারি 2L' = ঠিক ২টা সিগন্যাল পরপর লসের পরের সিগন্যালের উইন রেট (সংখ্যা = কতবার ঘটেছে)। "
             "EV = প্রথম স্টেকের ইউনিটে প্রতি সিগন্যালে গড় লাভ/লস (পেআউট %d%%, MTG গুণ %.1f ধরে)। সময় UTC+6 ফিক্সড।" % (round(E2_PAYOUT * 100), E2_MULT))


def e2_text():
    side = E2["side"]
    mtg = {"NON": "No-MTG", "MTG1": "MTG1", "BOTH": "No-MTG + MTG1"}[E2["mtg"]]
    return ("🔬 ইঞ্জিন ২ — Back Signal Discovery\n\n"
            "সব ২০ পেয়ার × প্রতি ঘণ্টার ০–৫৯ মিনিট × UP/DOWN × MTG। প্রথম ২/৩ অংশে প্যাটার্ন খোঁজে, শেষ ১/৩ অংশে (OOS) যাচাই করে, "
            "প্রতি মাসের স্থিতিশীলতা দেখে, ধারাবাহিক সিগন্যাল-লস (দিন ভাঙে না) গোনে।\n\n"
            "⚙️ এখনকার সেটিং:\n• সময়কাল: %d মাস\n• দিক: %s\n• MTG: %s\n• ন্যূনতম অ্যাকুরেসি: %d%%\n• ন্যূনতম নমুনা (ট্রেইন অংশে): %d\n\n"
            "নির্দিষ্ট পেয়ার চাইলে: /engine2 USDJPY EURJPY\nসেটিং বদলে ▶️ চালাও চাপো।" % (
                E2["months"], side, mtg, round(E2["acc"] * 100), E2["nmin"]))


def e2_kb():
    def row(label, key, vals, fmt):
        return [Btn(("✔ " if E2[key] == v else "") + fmt(v), callback_data="e2:%s:%s" % (key, v)) for v in vals]
    rows = [row("m", "months", (3, 4, 6, 9, 12), lambda v: "%dM" % v),
            row("s", "side", ("UP", "DOWN", "BOTH"), str),
            row("g", "mtg", ("NON", "MTG1", "BOTH"), lambda v: {"NON": "No-MTG", "MTG1": "MTG1", "BOTH": "দুটোই"}[v]),
            row("a", "acc", (0.6, 0.7, 0.8), lambda v: "%d%%" % round(v * 100)),
            row("a", "acc", (0.85, 0.9, 0.95), lambda v: "%d%%" % round(v * 100)),
            row("n", "nmin", (100, 300, 1000), lambda v: "নমুনা≥%d" % v),
            [Btn("▶️ চালাও (সব পেয়ার)", callback_data="e2:run:0")]]
    return Markup(rows)


def fmt2(i, r):
    mtg = "No-MTG" if r["mtg"] == 0 else "MTG%d" % r["mtg"]
    rec = " ".join("%dL→%s(%d)" % (N + 1, pct(r["rec_w"][N], r["rec_n"][N]), r["rec_n"][N]) for N in range(4))
    return ("%s%d. %s | :%02d %s | %s\n"
            "    ট্রেইন %s (%d) • OOS %s (%d) • মোট %s (%d/%d)\n"
            "    Normal %s • MTG-জয় %d • লস %d • ডোজি %d • সর্বোচ্চ লস %d\n"
            "    রিকভারি: %s\n"
            "    মাস %d/%d ভালো • EV %+.2f" % (
                E2_ICON[r["status"]], i, pairs.pretty(r["pair"]), r["minute"], r["side"], mtg,
                pct(r["wtr"], r["ntr"]), r["ntr"], pct(r["wte"], r["nte"]), r["nte"], pct(r["w"], r["n"]), r["w"], r["n"],
                pct(r["nw"], r["n"]), r["mw"], r["loss"], r["doji"], r["maxl"], rec, r["months_ok"], r["months_tot"], r["ev"]))


def e2_csv(rows, months):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["status", "pair", "minute", "side", "mtg", "period_months", "total_signals", "normal_win", "mtg_win", "final_win",
                "final_loss", "doji", "normal_acc_%", "final_acc_%", "coin_flip_%", "train_acc_%", "train_n", "oos_acc_%", "oos_n",
                "max_consec_loss", "rec1_acc_%", "rec1_n", "rec2_acc_%", "rec2_n", "rec3_acc_%", "rec3_n", "rec4_acc_%", "rec4_n",
                "rec5_acc_%", "rec5_n", "months_good", "months_total", "folds_ok", "EV_per_signal"])
    P = lambda a, b: round(100.0 * a / b, 1) if b else ""   # noqa
    for r in rows:
        line = [r["status"], pairs.pretty(r["pair"]), r["minute"], r["side"], r["mtg"], months, r["n"], r["nw"], r["mw"], r["w"], r["loss"],
                r["doji"], P(r["nw"], r["n"]), P(r["w"], r["n"]), round(100 * r["base"], 1), P(r["wtr"], r["ntr"]), r["ntr"],
                P(r["wte"], r["nte"]), r["nte"], r["maxl"]]
        for N in range(engine2.NREC):
            line += [P(r["rec_w"][N], r["rec_n"][N]), r["rec_n"][N]]
        line += [r["months_ok"], r["months_tot"], "yes" if r["fold_ok"] else "no", round(r["ev"], 3)]
        w.writerow(line)
    return io.BytesIO(buf.getvalue().encode("utf-8-sig"))


async def do_engine2(progress, only=None):
    """-> (text, csv BytesIO). Reads candles read-only, nothing is written to any database."""
    if not source.enabled():
        raise RuntimeError("DATABASE_URL_1/2/3 সেট করা নেই (লাইভ বটের ক্যান্ডেল ডেটাবেজ)।")
    async with SCAN_LOCK:       # same lock as /scan: two heavy jobs at once would eat Render's small RAM
        t0 = time.time()
        plist = only or pairs.PAIRS
        sides = ["UP", "DOWN"] if E2["side"] == "BOTH" else [E2["side"]]
        mtgs = {"NON": [0], "MTG1": [1], "BOTH": [0, 1]}[E2["mtg"]]
        n_tests = len(plist) * 60 * len(sides) * len(mtgs)
        z = E2_Z or engine2.bonferroni_z(n_tests)
        p = dict(min_acc=E2["acc"], min_n=E2["nmin"], min_test_n=max(30, E2["nmin"] // 3), z=z, split=PARAMS["split"],
                 sides=sides, mtgs=mtgs, minutes=range(60), payout=E2_PAYOUT, mult=E2_MULT)
        now = int(time.time())
        allrows, quals, missing = [], [], []
        for i, pair in enumerate(plist):
            await progress("⏳ ইঞ্জিন ২: %d/%d • %s" % (i + 1, len(plist), pairs.pretty(pair)))
            ts, d = await asyncio.to_thread(source.load_dir, pair, now - E2["months"] * 30 * 86400, now + 86400)
            if len(ts) < 5000:
                missing.append(pairs.pretty(pair))
                continue
            rows, q = await asyncio.to_thread(engine2.scan_pair, pair, ts, d, p)
            allrows += rows
            quals.append(q)
    engine2.sort_rows(allrows)
    cnt = {k: sum(1 for r in allrows if r["status"] == k) for k in ("valid", "low", "overfit", "no")}
    valid = engine.dedupe([r for r in allrows if r["status"] == "valid"])
    txt = "🔬 ইঞ্জিন ২ শেষ (%d সেকেন্ড)\n%d পেয়ার • %d মাসের ডেটা চাওয়া হয়েছে • %d টা স্লট পরীক্ষা (সব ৬০ মিনিট)\n" % (
        int(time.time() - t0), len(quals), E2["months"], len(allrows))
    if quals:
        txt += "ডেটার মান: গড় কভারেজ %d%% (বাজার-খোলা দিনে) • ডোজি %.2f%% • গড় %d ট্রেডিং-দিন\n" % (
            round(100 * sum(q["coverage"] for q in quals) / len(quals)), 100 * sum(q["doji"] for q in quals) / len(quals),
            sum(q["days"] for q in quals) // len(quals))
    if missing:
        txt += "ডেটা নেই/কম: %s\n" % ", ".join(missing)
    txt += ("কড়াকড়ি: z = %.2f (%d টা টেস্ট ধরে ভুয়া বাগ আটকাতে) • ন্যূনতম WR %d%% • নমুনা ≥ %d\n"
            "✅ %d • 🟡 %d • ❌ ওভারফিট %d • বাকি (প্রার্থী না) %d\n" % (
                z, n_tests, round(E2["acc"] * 100), E2["nmin"], cnt["valid"], cnt["low"], cnt["overfit"], cnt["no"]))
    if valid:
        txt += "\nসেরা ✅ বাগ (একই বাগের ভ্যারিয়েন্ট এক ধরে):\n" + "\n".join(fmt2(i + 1, r) for i, r in enumerate(valid[:8]))
    else:
        txt += "\nএই শর্তে কোনো ✅ বাগ নেই — এটা অস্বাভাবিক কিছু না, সত্যিকারের বাগ খুব বিরল।"
        near = [r for r in allrows if r["status"] != "no"][:5] or allrows[:5]
        if near:
            txt += "\nসবচেয়ে কাছাকাছি ৫টা (বাগ না, শুধু কাছাকাছি):\n" + "\n".join(fmt2(i + 1, r) for i, r in enumerate(near))
    txt += "\n\n" + E2_LEGEND
    return txt, e2_csv(allrows, E2["months"])


async def e2_job(chat_id, prog, only=None):
    async def progress(t):
        try:
            await prog.edit_text(t)
        except BadRequest:
            pass
    try:
        txt, data = await do_engine2(progress, only)
    except Exception as e:  # noqa
        log.exception("engine2 failed")
        return await progress("❌ ইঞ্জিন ২ হয়নি: %s" % e)
    try:
        await prog.delete()
    except Exception:  # noqa
        pass
    for part in chunks(txt):
        await send(chat_id, part)
    data.name = "engine2_%s.csv" % dt.datetime.now(pairs.BDT).strftime("%Y%m%d_%H%M")
    try:
        await APP.bot.send_document(chat_id, data, caption="ইঞ্জিন ২: সব স্লটের পূর্ণ তালিকা (✅ আগে)")
    except Exception:  # noqa
        log.exception("engine2 csv send")


async def cmd_engine2(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return await deny(update)
    msg = update.effective_message
    only = [p for p in (pairs.norm(t) for a in (ctx.args or []) for t in a.replace(",", " ").split()) if p]
    if only:
        if SCAN_LOCK.locked():
            return await msg.reply_text("একটা স্ক্যান আগে থেকেই চলছে, শেষ হলে আবার চেষ্টা করো।")
        prog = await msg.reply_text("⏳ ইঞ্জিন ২ শুরু…")
        ctx.application.bot_data["e2_job"] = asyncio.create_task(e2_job(msg.chat_id, prog, only))
        return
    await msg.reply_text(e2_text(), reply_markup=e2_kb())


async def on_e2_cb(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not allowed(update):
        return await q.answer("অনুমতি নেই", show_alert=True)
    _, key, val = q.data.split(":")
    if key == "run":
        if SCAN_LOCK.locked():
            return await q.answer("একটা স্ক্যান চলছে, শেষ হলে চাপো।", show_alert=True)
        await q.answer("শুরু হচ্ছে…")
        prog = await q.message.reply_text("⏳ ইঞ্জিন ২ শুরু…")
        ctx.application.bot_data["e2_job"] = asyncio.create_task(e2_job(q.message.chat_id, prog))
        return
    await q.answer()
    E2[key] = {"months": int, "nmin": int, "acc": float}.get(key, str)(val)
    try:
        await q.edit_message_text(e2_text(), reply_markup=e2_kb())
    except BadRequest:
        pass


# ---------------------------------------------------------------- /engine3  (Minute Grid: G/R table + win ratio, PDF / CSV)
E3 = dict(tf=1, months=1, out="PDF", pairs=[], minutes=[], page="main")
E3_MONTHS = {1: "1M", 2: "2M", 3: "3M", 6: "6M", 12: "1Y"}
E3_OUT = {"PDF": "PDF", "CSV": "CSV", "BOTH": "দুটোই"}


def e3_text():
    ps = ", ".join(pairs.pretty(p) for p in E3["pairs"]) or "— (বাছাই করোনি)"
    ms = " ".join(":%02d" % m for m in sorted(E3["minutes"])) or "— (বাছাই করোনি)"
    return ("🧮 ইঞ্জিন ৩ — মিনিট গ্রিড\n\n"
            "প্রতি ঘণ্টার (১২টা–২৩টা, UTC+6) বাছাই করা মিনিটের ক্যান্ডেল G/R টেবিলে সাজায়, নন-MTG ও MTG1 উইনিং রেশিও এবং ব্যাক-টু-ব্যাক লস বের করে।\n\n"
            "⚙️ এখনকার সেটিং:\n• টাইম ফ্রেম: %d মিনিট\n• সময়কাল: %s\n• আউটপুট: %s\n• পেয়ার: %s\n• মিনিট: %s\n\n"
            "পেয়ার ও মিনিট বাছাই করে ▶️ চালাও চাপো। (/engine3 USDJPY EURJPY লিখলে পেয়ার আগে থেকেই বাছা হবে)" % (
                E3["tf"], E3_MONTHS[E3["months"]], E3_OUT[E3["out"]], ps, ms))


def e3_kb():
    def tick(on, label):
        return ("✔ " if on else "") + label
    if E3["page"] == "pairs":
        rows, cur = [], []
        for p in pairs.PAIRS:
            cur.append(Btn(tick(p in E3["pairs"], pairs.pretty(p)), callback_data="e3:tp:%s" % p))
            if len(cur) == 3:
                rows.append(cur)
                cur = []
        if cur:
            rows.append(cur)
        rows.append([Btn("সব পেয়ার", callback_data="e3:pall:0"), Btn("পরিষ্কার", callback_data="e3:pclr:0"), Btn("⬅️ ফিরে", callback_data="e3:pg:main")])
        return Markup(rows)
    if E3["page"] == "minutes":
        rows, cur = [], []
        for m in range(0, 60, E3["tf"]):
            cur.append(Btn(tick(m in E3["minutes"], "%02d" % m), callback_data="e3:tm:%d" % m))
            if len(cur) == 6:
                rows.append(cur)
                cur = []
        if cur:
            rows.append(cur)
        rows.append([Btn("সব মিনিট", callback_data="e3:mall:0"), Btn("পরিষ্কার", callback_data="e3:mclr:0"), Btn("⬅️ ফিরে", callback_data="e3:pg:main")])
        return Markup(rows)
    return Markup([
        [Btn(tick(E3["tf"] == v, "%d মিনিট" % v), callback_data="e3:tf:%d" % v) for v in (1, 3, 5)],
        [Btn(tick(E3["months"] == v, lb), callback_data="e3:mo:%d" % v) for v, lb in E3_MONTHS.items()],
        [Btn(tick(E3["out"] == v, lb), callback_data="e3:out:%s" % v) for v, lb in E3_OUT.items()],
        [Btn("💱 পেয়ার (%d)" % len(E3["pairs"]), callback_data="e3:pg:pairs"), Btn("⏱ মিনিট (%d)" % len(E3["minutes"]), callback_data="e3:pg:minutes")],
        [Btn("▶️ চালাও", callback_data="e3:run:0")]])


async def e3_job(chat_id, prog, st):
    async def progress(t):
        try:
            await prog.edit_text(t)
        except BadRequest:
            pass
    if not source.enabled():
        return await progress("❌ DATABASE_URL_1/2/3 সেট করা নেই (লাইভ বটের ক্যান্ডেল ডেটাবেজ)।")
    days = engine3.PERIOD_DAYS[st["months"]]
    stamp = dt.datetime.now(pairs.BDT).strftime("%Y%m%d_%H%M")
    missing = []
    try:
        async with SCAN_LOCK:       # same lock as /scan and /engine2 (Render's small RAM)
            for i, pair in enumerate(st["pairs"]):
                await progress("⏳ ইঞ্জিন ৩: %d/%d • %s" % (i + 1, len(st["pairs"]), pairs.pretty(pair)))
                now = int(time.time())
                ts, o, c = await asyncio.to_thread(source.load_oc, pair, now - (days + 10) * 86400, now + 86400)
                res = await asyncio.to_thread(engine3.analyze, pair, ts, o, c, st["tf"], st["months"], st["minutes"])
                if res is None:
                    missing.append(pairs.pretty(pair))
                    continue
                tag = "%s_M%d_%s" % (pair, st["tf"], stamp)
                head = "📊 %s • %d মিনিট • %s • %d ট্রেডিং-দিন (%s → %s)" % (
                    pairs.pretty(pair), st["tf"], E3_MONTHS[st["months"]], len(res["days"]),
                    engine3.day_date(res["d0"], res["days"][0]), engine3.day_date(res["d0"], res["days"][-1]))
                body = engine3.bn_text(res)
                for part in chunks(head + "\n\n" + body):
                    await send(chat_id, part)
                if st["out"] in ("PDF", "BOTH"):
                    await progress("⏳ PDF বানাচ্ছি: %s" % pairs.pretty(pair))
                    bio = await asyncio.to_thread(engine3.make_pdf, res)
                    bio.name = "engine3_%s.pdf" % tag
                    await APP.bot.send_document(chat_id, bio, caption=head[:1000])
                if st["out"] in ("CSV", "BOTH"):
                    g, s = await asyncio.to_thread(engine3.make_csv, res)
                    await APP.bot.send_document(chat_id, io.BytesIO(g), filename="engine3_grid_%s.csv" % tag, caption="G/R গ্রিড (প্রতি দিন × ঘণ্টা × মিনিট)")
                    await APP.bot.send_document(chat_id, io.BytesIO(s), filename="engine3_summary_%s.csv" % tag, caption="উইন রেশিও ও ব্যাক-টু-ব্যাক লস")
    except Exception as e:  # noqa
        log.exception("engine3 failed")
        return await progress("❌ ইঞ্জিন ৩ হয়নি: %s" % e)
    try:
        await prog.delete()
    except Exception:  # noqa
        pass
    if missing:
        await send(chat_id, "ডেটা নেই/কম: %s" % ", ".join(missing))
    await send(chat_id, "✅ ইঞ্জিন ৩ শেষ।")


async def cmd_engine3(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return await deny(update)
    only = [p for p in (pairs.norm(t) for a in (ctx.args or []) for t in a.replace(",", " ").split()) if p]
    if only:
        E3["pairs"] = list(dict.fromkeys(only))
    E3["page"] = "main"
    await update.effective_message.reply_text(e3_text(), reply_markup=e3_kb())


async def on_e3_cb(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not allowed(update):
        return await q.answer("অনুমতি নেই", show_alert=True)
    _, key, val = q.data.split(":", 2)
    if key == "run":
        if not E3["pairs"] or not E3["minutes"]:
            return await q.answer("আগে অন্তত একটা পেয়ার আর একটা মিনিট বাছো", show_alert=True)
        if SCAN_LOCK.locked():
            return await q.answer("একটা স্ক্যান চলছে, শেষ হলে চাপো।", show_alert=True)
        await q.answer("শুরু হচ্ছে…")
        prog = await q.message.reply_text("⏳ ইঞ্জিন ৩ শুরু…")
        st = dict(E3, pairs=list(E3["pairs"]), minutes=sorted(E3["minutes"]))
        ctx.application.bot_data["e3_job"] = asyncio.create_task(e3_job(q.message.chat_id, prog, st))
        return
    await q.answer()
    if key == "pg":
        E3["page"] = val
    elif key == "tf":
        E3["tf"] = int(val)
        E3["minutes"] = [m for m in E3["minutes"] if m % E3["tf"] == 0]
    elif key == "mo":
        E3["months"] = int(val)
    elif key == "out":
        E3["out"] = val
    elif key == "tp":
        E3["pairs"] = [p for p in E3["pairs"] if p != val] if val in E3["pairs"] else E3["pairs"] + [val]
    elif key == "pall":
        E3["pairs"] = list(pairs.PAIRS)
    elif key == "pclr":
        E3["pairs"] = []
    elif key == "tm":
        m = int(val)
        E3["minutes"] = [x for x in E3["minutes"] if x != m] if m in E3["minutes"] else E3["minutes"] + [m]
    elif key == "mall":
        E3["minutes"] = list(range(0, 60, E3["tf"]))
    elif key == "mclr":
        E3["minutes"] = []
    try:
        await q.edit_message_text(e3_text(), reply_markup=e3_kb())
    except BadRequest:
        pass


# ---------------------------------------------------------------- background
async def auto_scan_loop(app):
    """Neon is read only when a scan is due (the next due time is kept in memory), so it can sleep in between."""
    await asyncio.sleep(90)
    due = 0.0
    try:
        if rdb.enabled():
            run = await asyncio.to_thread(rdb.latest_run)
            due = (run[2] if run else 0) + AUTO_SCAN_HOURS * 3600
    except Exception:  # noqa
        log.exception("auto scan: first check")
    while True:
        try:
            if source.enabled() and rdb.enabled() and not SCAN_LOCK.locked() and time.time() >= due:
                async def quiet(_t):
                    return None
                _, txt, new_txt = await do_scan(quiet)
                due = time.time() + AUTO_SCAN_HOURS * 3600
                for uid in ALLOWED:
                    await send(uid, "🔄 অটো স্ক্যান\n" + txt[:3500])
                    if new_txt:
                        await send(uid, new_txt[:3900])
        except Exception:  # noqa
            log.exception("auto scan")
            due = time.time() + 3600  # try again in an hour
        await asyncio.sleep(600)


async def post_init(app):
    global APP
    APP = app
    cmds = [BotCommand("scan", "এখনই বাগ স্ক্যান"), BotCommand("bugs", "পাওয়া বাগের তালিকা"), BotCommand("csv", "তালিকা CSV"), BotCommand("engine2", "ইঞ্জিন ২: Back Signal Discovery"),
            BotCommand("engine3", "ইঞ্জিন ৩: মিনিট গ্রিড (G/R টেবিল, PDF/CSV)"),
            BotCommand("live", "লাইভ অ্যালার্ট"), BotCommand("status", "অবস্থা"), BotCommand("data", "ডেটা কভারেজ"), BotCommand("check", "একটা স্লটের হিসাব দেখো"),
            BotCommand("id", "আমার Telegram ID")]
    try:
        await app.bot.set_my_commands(cmds)
    except Exception:  # noqa
        log.exception("set_my_commands")
    if rdb.enabled():
        try:
            await asyncio.to_thread(rdb.init)
            await live.load_state()
        except Exception:  # noqa
            log.exception("results database init failed")
    else:
        log.error("RESULTS_DATABASE_URL missing")
    if not source.enabled():
        log.error("DATABASE_URL_* missing")
    app.bot_data["alert_task"] = asyncio.create_task(live.alert_loop(send))
    app.bot_data["result_task"] = asyncio.create_task(live.result_loop(send))
    app.bot_data["auto_scan_task"] = asyncio.create_task(auto_scan_loop(app))


class Health(BaseHTTPRequestHandler):
    def _r(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()

    def do_GET(self):
        self._r()
        self.wfile.write(b"ok")

    def do_HEAD(self):
        self._r()

    def log_message(self, *a):
        pass


class Quiet(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        pass


def serve():
    Quiet(("0.0.0.0", int(os.environ.get("PORT", "10000"))), Health).serve_forever()


def main():
    threading.Thread(target=serve, daemon=True).start()
    app = Application.builder().token(BOT_TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("scan", cmd_scan))
    app.add_handler(CommandHandler("bugs", cmd_bugs))
    app.add_handler(CommandHandler("csv", cmd_csv))
    app.add_handler(CommandHandler("live", cmd_live))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("data", cmd_data))
    app.add_handler(CommandHandler("check", cmd_check))
    app.add_handler(CommandHandler("engine2", cmd_engine2))
    app.add_handler(CommandHandler("engine3", cmd_engine3))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(CallbackQueryHandler(on_bugs_cb, pattern=r"^bg:"))
    app.add_handler(CallbackQueryHandler(on_live_cb, pattern=r"^lv:"))
    app.add_handler(CallbackQueryHandler(on_e2_cb, pattern=r"^e2:"))
    app.add_handler(CallbackQueryHandler(on_e3_cb, pattern=r"^e3:"))
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()

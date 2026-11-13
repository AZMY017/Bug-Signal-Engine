"""Engine 4 — Loss-Streak Signal Finder (pure numpy + reportlab, no Telegram, no database). engine.py / engine2.py / engine3.py are untouched.

Question it answers (one pair, one timeframe, last N months, chosen minute slot, chosen colour):
  "Every hour 12:00..23:59 (UTC+6) I trade the same minute (e.g. :05) for GREEN (or RED). After N back-to-back losses,
   how many times did the NEXT trade come up in the period, and how many of those won / lost?"

Rules (candle / MTG rules as Engine 3, but DOJI = WIN here):
 * Trade result per hour: win / loss. MTG k: entry goes your way = win; else the next candles may win (MTG1 = 2nd candle,
   MTG2 = 3rd, MTG3 = 4th); all k+1 candles wrong = loss.
 * DOJI (open == close) counts as a WIN (entry or any MTG step). The result tells how many of the wins were doji wins.
 * Only a missing candle (no data) gives no result: it is skipped and does not change the streak.
 * Streak = consecutive losses in hour order. scope "day": restarts every day. scope "chain": continues from one day into the next.
 * SIGNAL = the first trade after the streak reached N ("exact": streak == N; "atleast": every trade made while streak >= N).
   The signal trade is then a win (streak resets) or a loss (streak grows).
"""
import csv
import io

import numpy as np

import engine3 as e3

HOURS = e3.HOURS
MAXK = e3.MAXK
MODES = e3.MODES
SIDE_NAME = {"G": "GREEN", "R": "RED"}


def outcomes(dirm, val, tf, minute, days, side, k):
    """-> (out (days,hours) int8: 1 win, -1 loss, 0 no data ; lvl int8: win level 0..k, -1 otherwise ; dj bool: the win came from a doji)."""
    per = 60 // tf
    S = 1440 // tf
    j = minute // tf
    s = 1 if side == "G" else -1
    fd = np.concatenate((dirm.reshape(-1), np.zeros(MAXK, np.int8))).astype(np.int8)
    fv = np.concatenate((val.reshape(-1), np.zeros(MAXK, bool)))
    idx = days[:, None] * S + np.array([h * per + j for h in HOURS])[None, :]
    e, ev = fd[idx], fv[idx]
    lvl = np.full(e.shape, -1, np.int8)
    lvl[ev & ((e == s) | (e == 0))] = 0
    dj = ev & (e == 0)
    alive = ev & (e == -s)
    for step in range(1, k + 1):
        x, xv = fd[idx + step], fv[idx + step]
        w = alive & xv & ((x == s) | (x == 0))
        dj |= w & (x == 0)
        lvl[w] = step
        alive = alive & xv & (x == -s)
    out = np.zeros(e.shape, np.int8)
    out[lvl >= 0] = 1
    out[alive] = -1
    return out, lvl, dj


def find_signals(out, lvl, dj, n_loss, scope, atleast):
    """-> (list of (day_row, hour_col, result 1/-1, win_level, doji_win bool), missing_count). missing = no-data cells where a signal was due."""
    nd, nh = out.shape
    sigs, missing = [], 0
    streak = 0
    for i in range(nd):
        if scope == "day":
            streak = 0
        for j in range(nh):
            v = int(out[i, j])
            due = streak >= n_loss if atleast else streak == n_loss
            if v == 0:
                if due:
                    missing += 1
                continue
            if due:
                sigs.append((i, j, v, int(lvl[i, j]), bool(dj[i, j])))
            streak = streak + 1 if v == -1 else 0
    return sigs, missing


def analyze(pair, ts, o, c, tf, months, minutes, sides, k, n_loss, scope, atleast):
    """-> dict or None when there is not enough data."""
    if len(ts) < 500:
        return None
    lt, d = e3.to_tf(ts, o, c, tf)
    if len(lt) == 0:
        return None
    d0, dirm, val = e3.build_grid(lt, d, tf)
    days = e3.pick_days(dirm, val, tf, months)
    if len(days) == 0:
        return None
    rows = []
    for m in sorted(minutes):
        for side in sides:
            out, lvl, dj = outcomes(dirm, val, tf, m, days, side, k)
            bw, bl = int((out == 1).sum()), int((out == -1).sum())
            sigs, miss = find_signals(out, lvl, dj, n_loss, scope, atleast)
            sw = sum(1 for x in sigs if x[2] == 1)
            sl = len(sigs) - sw
            levels = [sum(1 for x in sigs if x[2] == 1 and x[3] == L) for L in range(k + 1)]
            rows.append(dict(minute=m, side=side, base_win=bw, base_loss=bl, base_dj=int((dj & (out == 1)).sum()),
                             base_wr=(bw / (bw + bl) if bw + bl else 0.0),
                             sig=len(sigs), win=sw, loss=sl, wr=(sw / len(sigs) if sigs else 0.0),
                             doji_win=sum(1 for x in sigs if x[2] == 1 and x[4]), missing=miss, levels=levels,
                             signals=[(int(days[i]), HOURS[j], v, L, d_) for i, j, v, L, d_ in sigs]))
    return dict(pair=pair, tf=tf, months=months, d0=d0, days=days, k=k, n_loss=n_loss, scope=scope, atleast=atleast, rows=rows)


def _mode_text(res):
    return "%s | %d back-to-back loss (%s, %s)" % (MODES[res["k"]], res["n_loss"], "N or more" if res["atleast"] else "exactly N",
                                                  "day" if res["scope"] == "day" else "chain")


# ------------------------------------------------------------------ CSV
def make_csv(res):
    """-> (summary_csv bytes, signals_csv bytes)."""
    s = io.StringIO()
    w = csv.writer(s)
    w.writerow(["pair", "tf_min", "mtg", "b2b_loss_n", "rule", "minute", "side", "days", "signals", "win", "of_which_doji_win", "loss", "winrate_pct",
                "win_direct", "win_mtg1", "win_mtg2", "win_mtg3", "no_data_skipped", "all_trades_win", "all_trades_loss", "all_trades_winrate_pct"])
    for r in res["rows"]:
        lv = r["levels"] + [""] * (MAXK + 1 - len(r["levels"]))
        w.writerow([res["pair"], res["tf"], MODES[res["k"]], res["n_loss"], ("atleast_" if res["atleast"] else "exact_") + res["scope"], "%02d" % r["minute"],
                    SIDE_NAME[r["side"]], len(res["days"]), r["sig"], r["win"], r["doji_win"], r["loss"], "%.1f" % (100 * r["wr"]),
                    lv[0], lv[1], lv[2], lv[3], r["missing"], r["base_win"], r["base_loss"], "%.1f" % (100 * r["base_wr"])])
    g = io.StringIO()
    w = csv.writer(g)
    w.writerow(["pair", "minute", "side", "date", "weekday", "hour", "result", "win_at", "doji_win"])
    for r in res["rows"]:
        for dayno, h, v, L, d_ in r["signals"]:
            dd = e3.day_date(res["d0"], dayno)
            w.writerow([res["pair"], "%02d" % r["minute"], SIDE_NAME[r["side"]], dd.isoformat(), e3.WD[dd.weekday()], "%02d" % h,
                        "WIN" if v == 1 else "LOSS", ("direct" if L == 0 else "MTG%d" % L) if v == 1 else "", "YES" if d_ else ""])
    return s.getvalue().encode("utf-8-sig"), g.getvalue().encode("utf-8-sig")


# ------------------------------------------------------------------ PDF
def make_pdf(res):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    k = res["k"]
    pair = res["pair"].replace("_", "/")
    GREEN, RED = colors.Color(0.0, 0.5, 0.25), colors.Color(0.8, 0.15, 0.2)
    NAVY, ZEBRA, LINE = colors.Color(0.09, 0.22, 0.4), colors.Color(0.94, 0.96, 0.98), colors.Color(0.8, 0.84, 0.9)
    page = landscape(A4)
    M = 24
    h1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=15, textColor=NAVY, spaceAfter=4)
    h2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=10.5, textColor=NAVY, spaceBefore=8, spaceAfter=3)
    p9 = ParagraphStyle("p9", fontName="Helvetica", fontSize=8.5, leading=11)
    dts = [e3.day_date(res["d0"], i) for i in res["days"]]
    story = [Paragraph("%s  -  Engine 4 Loss-Streak Signals  (M%d)" % (pair, res["tf"]), h1)]
    story.append(Paragraph("Period: %s to %s (%d trading days, last %s)  |  Entry hours 12:00-23:59 BDT (UTC+6)  |  %s" % (
        dts[0], dts[-1], len(dts), {1: "1 month", 2: "2 months", 3: "3 months", 6: "6 months", 12: "1 year"}[res["months"]], _mode_text(res)), p9))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "<b>How to read:</b> every hour you trade the chosen minute (GREEN or RED). A <b>signal</b> is the trade that comes right after the chosen number of "
        "back-to-back losses. <b>Win breakdown</b> = wins that came directly, then from MTG step 1, 2, 3. "
        "<b>Doji rule:</b> a doji candle (open = close) is counted as a WIN (entry or MTG step); the column <b>Doji wins</b> says how many of the signal wins were doji wins. "
        "<b>No data</b> = a candle was missing, nothing counted. <b>All trades</b> = win rate of that same slot without any streak condition, for comparison.", p9))
    head = ["Minute", "Trade", "Signals", "Win", "of which doji", "Loss", "WR %", "Win breakdown", "No data", "All trades WR %"]
    rows = [head]
    style = [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
             ("FONTSIZE", (0, 0), (-1, -1), 8), ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("GRID", (0, 0), (-1, -1), 0.4, LINE)]
    for r in res["rows"]:
        bd = " + ".join(str(v) for v in r["levels"])
        rows.append([":%02d" % r["minute"], "GREEN" if r["side"] == "G" else "RED", r["sig"], r["win"], r["doji_win"], r["loss"],
                     "%.1f" % (100 * r["wr"]) if r["sig"] else "-", bd, r["missing"], "%.1f (%d/%d)" % (100 * r["base_wr"], r["base_win"], r["base_loss"])])
        i = len(rows) - 1
        style.append(("TEXTCOLOR", (1, i), (1, i), GREEN if r["side"] == "G" else RED))
        style.append(("FONTNAME", (1, i), (1, i), "Helvetica-Bold"))
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    t = Table(rows, repeatRows=1, colWidths=[50, 55, 55, 45, 70, 45, 50, 110, 55, 110])
    t.setStyle(TableStyle(style))
    story += [Paragraph("Summary", h2), t]
    for r in res["rows"]:
        if not r["signals"]:
            continue
        story.append(PageBreak())
        story.append(Paragraph("Signals  -  minute :%02d  %s  (%d signals: %d win, %d loss, %d of the wins are doji)" % (
            r["minute"], "GREEN" if r["side"] == "G" else "RED", r["sig"], r["win"], r["loss"], r["doji_win"]), h2))
        data = [["#", "Date", "Day", "Hour", "Result", "Won at", "Doji win"]]
        st2 = [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
               ("FONTSIZE", (0, 0), (-1, -1), 7.6), ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("GRID", (0, 0), (-1, -1), 0.4, LINE),
               ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]
        for n, (dayno, h, v, L, d_) in enumerate(r["signals"], 1):
            dd = e3.day_date(res["d0"], dayno)
            data.append([n, dd.isoformat(), e3.WD[dd.weekday()], "%02d:%02d" % (h, r["minute"]), "WIN" if v == 1 else "LOSS",
                         ("direct" if L == 0 else "MTG%d" % L) if v == 1 else "-", "YES" if d_ else ""])
            st2.append(("TEXTCOLOR", (4, n), (4, n), GREEN if v == 1 else RED))
            st2.append(("FONTNAME", (4, n), (4, n), "Helvetica-Bold"))
            if n % 2 == 0:
                st2.append(("BACKGROUND", (0, n), (-1, n), ZEBRA))
        t2 = Table(data, repeatRows=1, colWidths=[35, 80, 45, 60, 60, 65, 60])
        t2.setStyle(TableStyle(st2))
        story.append(t2)
    bio = io.BytesIO()
    SimpleDocTemplate(bio, pagesize=page, leftMargin=M, rightMargin=M, topMargin=M, bottomMargin=M,
                      title="%s M%d engine4" % (pair, res["tf"])).build(story)
    bio.seek(0)
    return bio


# ------------------------------------------------------------------ Telegram text (Bengali)
def bn_text(res, limit=24):
    k = res["k"]
    out = ["ℹ️ ডোজি = উইন হিসেবে গোনা হয়েছে।", ""]
    for r in res["rows"][:limit]:
        icon = "🟢" if r["side"] == "G" else "🔴"
        nm = "গ্রিন" if r["side"] == "G" else "রেড"
        out.append("%s :%02d %s" % (icon, r["minute"], nm))
        if r["sig"] == 0:
            out.append("   এই সময়ে কোনো সিগন্যাল আসেনি")
        else:
            out.append("   📡 সিগন্যাল %d • ✅ উইন %d • ❌ লস %d • উইন রেট %.0f%%" % (r["sig"], r["win"], r["loss"], 100 * r["wr"]))
            if k:
                parts = ["ডাইরেক্ট %d" % r["levels"][0]] + ["MTG%d %d" % (L, r["levels"][L]) for L in range(1, k + 1)]
                out.append("   উইন ভাঙা: " + " + ".join(parts))
            out.append("   🟰 %d টা উইনের মধ্যে %d টা ডোজি (ডোজিকে উইন ধরা হয়েছে)" % (r["win"], r["doji_win"]))
        if r["missing"]:
            out.append("   ➖ ক্যান্ডেল ডেটা নেই (স্কিপ): %d" % r["missing"])
        out.append("   তুলনা: এই স্লটের সব ট্রেডে উইন রেট %.0f%% (%d উইন / %d লস)" % (100 * r["base_wr"], r["base_win"], r["base_loss"]))
    if len(res["rows"]) > limit:
        out.append("… বাকিগুলো PDF/CSV-তে।")
    return "\n".join(out)

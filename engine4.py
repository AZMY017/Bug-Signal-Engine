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
import os

FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "FreeSans.ttf")   # GNU FreeSans (has Bengali)
WD_BN = ["সোম", "মঙ্গল", "বুধ", "বৃহ", "শুক্র", "শনি", "রবি"]


def bn_pdf_ok():
    """Bengali PDF needs the font file AND Pillow's text shaping (raqm) - reportlab alone cannot join Bengali letters."""
    try:
        from PIL import features
        return os.path.exists(FONT_PATH) and bool(features.check("raqm"))
    except Exception:  # noqa
        return False


def make_pdf(res, lang="auto"):
    """lang 'bn' = Bengali text (rendered with shaping as images), 'en' = English, 'auto' = Bengali when possible."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    bn = bn_pdf_ok() if lang == "auto" else (lang == "bn" and bn_pdf_ok())
    res["pdf_lang"] = "bn" if bn else "en"
    SC = 4.0   # image pixels per point

    def rgb(c):
        return tuple(int(round(255 * v)) for v in (c.red, c.green, c.blue))

    def T(en, bnt, size=8, color=colors.black, bold=False, maxw=None, align="left"):
        """text cell/flowable: English string in English mode, shaped Bengali image in Bengali mode."""
        if not bn:
            return en
        from PIL import Image, ImageDraw, ImageFont
        f = ImageFont.truetype(FONT_PATH, int(size * SC), layout_engine=ImageFont.Layout.RAQM)
        lines, cur = [], ""
        for w_ in bnt.split(" "):
            t_ = (cur + " " + w_).strip()
            if maxw and cur and f.getlength(t_) > maxw * SC:
                lines.append(cur)
                cur = w_
            else:
                cur = t_
        lines.append(cur)
        lead = int(size * SC * 1.55)
        wpx = max(int(f.getlength(x)) for x in lines) + 6
        hpx = lead * len(lines) + int(size * SC * 0.35)
        im = Image.new("RGBA", (wpx, hpx), rgb(color) + (0,))
        d = ImageDraw.Draw(im)
        for n, x in enumerate(lines):
            d.text((3, n * lead + int(size * SC * 0.2)), x, font=f, fill=rgb(color) + (255,), stroke_width=(1 if bold else 0), stroke_fill=rgb(color) + (255,))
        bio_ = io.BytesIO()
        im.save(bio_, "PNG")
        bio_.seek(0)
        img = RLImage(bio_, width=wpx / SC, height=hpx / SC)
        img.hAlign = "LEFT" if align == "left" else "CENTER"
        return img

    pair = res["pair"].replace("_", "/")
    GREEN, RED = colors.Color(0.0, 0.5, 0.25), colors.Color(0.8, 0.15, 0.2)
    NAVY, ZEBRA, LINE = colors.Color(0.09, 0.22, 0.4), colors.Color(0.94, 0.96, 0.98), colors.Color(0.8, 0.84, 0.9)
    page = landscape(A4)
    M = 24
    width = page[0] - 2 * M
    h1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=15, textColor=NAVY, spaceAfter=4)
    p9 = ParagraphStyle("p9", fontName="Helvetica", fontSize=8.5, leading=11)

    def flow(en, bnt, size=8.5, color=colors.black, bold=False, style=p9):
        if bn:
            return T(en, bnt, size, color, bold, maxw=width)
        return Paragraph(en, style)

    dts = [e3.day_date(res["d0"], i) for i in res["days"]]
    per_en = {1: "1 month", 2: "2 months", 3: "3 months", 6: "6 months", 12: "1 year"}[res["months"]]
    per_bn = {1: "১ মাস", 2: "২ মাস", 3: "৩ মাস", 6: "৬ মাস", 12: "১ বছর"}[res["months"]]
    mtg_bn = MODES[res["k"]].replace("No-MTG", "নন-MTG")
    rule_en = _mode_text(res)
    rule_bn = "%s | পরপর %d লস (%s, %s)" % (mtg_bn, res["n_loss"], "N বা বেশি" if res["atleast"] else "ঠিক N", "ডে" if res["scope"] == "day" else "চেইন")

    story = [flow("%s  -  Engine 4 Loss-Streak Signals  (M%d)" % (pair, res["tf"]),
                  "%s  -  ইঞ্জিন ৪: লসের পর সিগন্যাল  (M%d)" % (pair, res["tf"]), 15, NAVY, True, h1)]
    story.append(flow("Period: %s to %s (%d trading days, last %s)  |  Entry hours 12:00-23:59 BDT (UTC+6)  |  %s" % (dts[0], dts[-1], len(dts), per_en, rule_en),
                      "সময়কাল: %s থেকে %s (%d ট্রেডিং-দিন, শেষ %s)  |  ঘণ্টা ১২:০০–২৩:৫৯ (UTC+6)  |  %s" % (dts[0], dts[-1], len(dts), per_bn, rule_bn), 8.5))
    story.append(Spacer(1, 3))
    story.append(flow(
        "<b>How to read:</b> every hour you trade the chosen minute (GREEN or RED). A <b>signal</b> is the trade that comes right after the chosen number of "
        "back-to-back losses. <b>Win breakdown</b> = wins that came directly, then from MTG step 1, 2, 3. "
        "<b>Doji rule:</b> a doji candle (open = close) is counted as a WIN (entry or MTG step); the column <b>Doji wins</b> says how many of the signal wins were doji wins. "
        "<b>No data</b> = a candle was missing, nothing counted. <b>All trades</b> = win rate of that same slot without any streak condition, for comparison.",
        "কীভাবে পড়বে: প্রতি ঘণ্টায় তুমি বাছা মিনিটে ট্রেড করো (গ্রিন বা রেড)। পরপর যতগুলো লস বেছেছ তার ঠিক পরের ট্রেডটাই সিগন্যাল। "
        "উইন ভাঙা = কয়টা উইন সরাসরি এসেছে, আর কয়টা MTG-র ১ম, ২য়, ৩য় ধাপে। "
        "ডোজির নিয়ম: ডোজি ক্যান্ডেল (ওপেন = ক্লোজ) উইন হিসেবে গোনা হয়েছে; 'তার মধ্যে ডোজি' কলামে দেখা যাবে সিগন্যালের উইনগুলোর মধ্যে কয়টা ডোজির কারণে উইন। "
        "ডেটা নেই = ক্যান্ডেল মিসিং, কিছু গোনা হয়নি। সব ট্রেডে উইন রেট = কোনো লস-শর্ত ছাড়া ওই একই স্লটের উইন রেট, তুলনার জন্য।", 8.5))
    story.append(Spacer(1, 5))
    story.append(flow("<b>Summary</b>", "সারসংক্ষেপ", 10.5, NAVY, True,
                      ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=10.5, textColor=NAVY, spaceAfter=3)))
    WH = colors.white
    head = [T("Minute", "মিনিট", 8, WH, True, align="c"), T("Trade", "ট্রেড", 8, WH, True, align="c"), T("Signals", "সিগন্যাল", 8, WH, True, align="c"),
            T("Win", "উইন", 8, WH, True, align="c"), T("of which doji", "তার মধ্যে ডোজি", 8, WH, True, align="c"), T("Loss", "লস", 8, WH, True, align="c"),
            T("WR %", "উইন রেট %", 8, WH, True, align="c"), T("Win breakdown", "উইন ভাঙা", 8, WH, True, align="c"),
            T("No data", "ডেটা নেই", 8, WH, True, align="c"), T("All trades WR %", "সব ট্রেডে উইন রেট %", 8, WH, True, align="c")]
    rows = [head]
    style = [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
             ("FONTSIZE", (0, 0), (-1, -1), 8), ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("GRID", (0, 0), (-1, -1), 0.4, LINE)]
    for r in res["rows"]:
        gcol = GREEN if r["side"] == "G" else RED
        rows.append([":%02d" % r["minute"],
                     T("GREEN" if r["side"] == "G" else "RED", "গ্রিন" if r["side"] == "G" else "রেড", 8, gcol, True, align="c"),
                     r["sig"], r["win"], r["doji_win"], r["loss"], "%.1f" % (100 * r["wr"]) if r["sig"] else "-",
                     " + ".join(str(v) for v in r["levels"]), r["missing"], "%.1f (%d/%d)" % (100 * r["base_wr"], r["base_win"], r["base_loss"])])
        i = len(rows) - 1
        if not bn:
            style.append(("TEXTCOLOR", (1, i), (1, i), gcol))
            style.append(("FONTNAME", (1, i), (1, i), "Helvetica-Bold"))
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    t = Table(rows, repeatRows=1, colWidths=[50, 55, 55, 45, 75, 45, 55, 110, 55, 115])
    t.setStyle(TableStyle(style))
    story.append(t)
    for r in res["rows"]:
        if not r["signals"]:
            continue
        story.append(PageBreak())
        side_en, side_bn = ("GREEN", "গ্রিন") if r["side"] == "G" else ("RED", "রেড")
        story.append(flow("<b>Signals - minute :%02d %s (%d signals: %d win, %d loss, %d of the wins are doji)</b>" % (r["minute"], side_en, r["sig"], r["win"], r["loss"], r["doji_win"]),
                          "সিগন্যালের তালিকা — মিনিট :%02d %s (%d সিগন্যাল: %d উইন, %d লস; উইনের মধ্যে %d টা ডোজি)" % (r["minute"], side_bn, r["sig"], r["win"], r["loss"], r["doji_win"]),
                          10.5, NAVY, True, ParagraphStyle("h2b", fontName="Helvetica-Bold", fontSize=10.5, textColor=NAVY, spaceAfter=4)))
        data = [["#", T("Date", "তারিখ", 7.6, WH, True, align="c"), T("Day", "বার", 7.6, WH, True, align="c"), T("Hour", "ঘণ্টা", 7.6, WH, True, align="c"),
                 T("Result", "ফলাফল", 7.6, WH, True, align="c"), T("Won at", "কোন ধাপে", 7.6, WH, True, align="c"), T("Doji win", "ডোজি উইন", 7.6, WH, True, align="c")]]
        st2 = [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
               ("FONTSIZE", (0, 0), (-1, -1), 7.6), ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("GRID", (0, 0), (-1, -1), 0.4, LINE),
               ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]
        for n, (dayno, h, v, L, d_) in enumerate(r["signals"], 1):
            dd = e3.day_date(res["d0"], dayno)
            col = GREEN if v == 1 else RED
            won_en = ("direct" if L == 0 else "MTG%d" % L) if v == 1 else "-"
            won_bn = ("সরাসরি" if L == 0 else "MTG%d" % L) if v == 1 else "-"
            data.append([n, dd.isoformat(), T(e3.WD[dd.weekday()], WD_BN[dd.weekday()], 7.6, align="c"), "%02d:%02d" % (h, r["minute"]),
                         T("WIN" if v == 1 else "LOSS", "উইন" if v == 1 else "লস", 7.6, col, True, align="c"), T(won_en, won_bn, 7.6, align="c"),
                         T("YES", "হ্যাঁ", 7.6, align="c") if d_ else ""])
            if not bn:
                st2.append(("TEXTCOLOR", (4, n), (4, n), col))
                st2.append(("FONTNAME", (4, n), (4, n), "Helvetica-Bold"))
            if n % 2 == 0:
                st2.append(("BACKGROUND", (0, n), (-1, n), ZEBRA))
        t2 = Table(data, repeatRows=1, colWidths=[35, 80, 50, 60, 60, 70, 65])
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

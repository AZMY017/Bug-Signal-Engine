"""Engine 5 — Ranked Minute Grid (pure numpy + reportlab, no Telegram, no database). engine.py .. engine4.py are untouched.

Engine 3's rules, statistics and candle logic are reused 1:1 (engine3.analyze), only the PDF / CSV / text layout is new:
 * One separate chart (table) per mode: No-MTG, then MTG1, MTG2, MTG3 (only up to the MTG step you choose).
 * Inside every chart the minutes are RANKED: the minute/side with the smallest back-to-back loss comes first
   (sort: B2B chain, then B2B day, then higher win rate, then earlier minute).
 * One row = one minute + one side (GREEN or RED), so you can read the best minutes from the top without hunting.
Data comes from the Neon candle database only (same as Engine 3).
"""
import csv
import io

import engine3 as e3

MAXK = e3.MAXK
MODES = e3.MODES
WD = e3.WD
SIDE = {"G": "GREEN", "R": "RED"}


def analyze(pair, ts, o, c, tf, months, minutes):
    """Same as Engine 3; empty `minutes` = every minute slot of the timeframe."""
    mins = sorted(minutes) if minutes else list(range(0, 60, tf))
    return e3.analyze(pair, ts, o, c, tf, months, mins)


def ranked(res, k, top=0):
    """-> list of row dicts for MTG level k, best first (smallest B2B loss). top=0 means all rows."""
    rows = []
    for st in res["stats"]:
        for side in ("G", "R"):
            x = st["sides"][side]["m"][k]
            rows.append(dict(minute=st["minute"], side=side, g=st["g"], r=st["r"], doji=st["doji"], days=st["days"],
                             win=x["win"], loss=x["loss"], wr=x["wr"], n=x["n"], levels=x["levels"],
                             skipped=x["skipped"], b2b_day=x["b2b_day"], b2b_chain=x["b2b_chain"],
                             day_at=x["b2b_day_at"], chain_at=x["b2b_chain_at"]))
    rows.sort(key=lambda r: (r["b2b_chain"], r["b2b_day"], -r["wr"], r["minute"], r["side"]))
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    return rows[:top] if top else rows


# ------------------------------------------------------------------ CSV
def make_csv(res, kmax=MAXK, top=0):
    """-> ranked CSV bytes: one block per mode, best minute first."""
    s = io.StringIO()
    w = csv.writer(s)
    w.writerow(["pair", "tf_min", "mode", "rank", "minute", "trade", "win", "loss", "winrate_pct", "win_direct",
                "win_mtg1", "win_mtg2", "win_mtg3", "step_skipped", "b2b_loss_day", "b2b_loss_chain",
                "b2b_loss_day_when", "b2b_loss_chain_when"])
    for k in range(kmax + 1):
        for r in ranked(res, k, top):
            lv = r["levels"] + [""] * (MAXK + 1 - len(r["levels"]))
            w.writerow([res["pair"], res["tf"], MODES[k], r["rank"], "%02d" % r["minute"], SIDE[r["side"]], r["win"],
                        r["loss"], "%.1f" % (100 * r["wr"]), lv[0], lv[1], lv[2], lv[3], r["skipped"], r["b2b_day"],
                        r["b2b_chain"], r["day_at"], r["chain_at"]])
    return s.getvalue().encode("utf-8-sig")


# ------------------------------------------------------------------ PDF
def make_pdf(res, kmax=MAXK, top=20):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    tf = res["tf"]
    pair = res["pair"].replace("_", "/")
    GREEN, RED = colors.Color(0.0, 0.5, 0.25), colors.Color(0.8, 0.15, 0.2)
    NAVY, ZEBRA, LINE = colors.Color(0.09, 0.22, 0.4), colors.Color(0.95, 0.97, 0.99), colors.Color(0.8, 0.84, 0.9)
    GOLD = colors.Color(1.0, 0.96, 0.78)
    page = landscape(A4)
    M = 20

    def b2b_bg(v):
        if v <= 2:
            return colors.Color(0.80, 0.93, 0.80)
        if v == 3:
            return colors.Color(0.90, 0.96, 0.78)
        if v == 4:
            return colors.Color(1.0, 0.95, 0.70)
        if v == 5:
            return colors.Color(1.0, 0.85, 0.65)
        return colors.Color(0.98, 0.72, 0.72)

    h1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=15, textColor=NAVY, spaceAfter=4)
    h2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12, textColor=NAVY, spaceBefore=4, spaceAfter=4)
    p9 = ParagraphStyle("p9", fontName="Helvetica", fontSize=8.5, leading=11)
    pc = ParagraphStyle("pc", fontName="Helvetica", fontSize=6.8, leading=8)

    dts = [e3.day_date(res["d0"], i) for i in res["days"]]
    story = [Paragraph("%s  -  Engine 5 Ranked Minute Chart  (M%d)" % (pair, tf), h1)]
    story.append(Paragraph(
        "Period: %s to %s  (%d trading days, last %s)   |   Entry hours 12:00-23:59 BDT (UTC+6)   |   Minutes checked: %d   |   Shown per chart: %s" % (
            dts[0], dts[-1], len(dts), e3.period_label_en(res["months"]), len(res["minutes"]),
            "top %d" % top if top else "all"), p9))
    story.append(Paragraph(
        "<b>How to read:</b> every mode has its own chart (No-MTG, MTG1, MTG2, MTG3 - up to the step you chose). "
        "Inside a chart the minutes are <b>ranked by the smallest back-to-back (B2B) loss first</b>: sorted by B2B chain, then B2B day, then higher win rate. "
        "One row = one minute + one side: <b>GREEN</b> = you trade GREEN at that minute every hour, <b>RED</b> = you trade RED. "
        "<b>MTG1</b> = the 2nd candle may win, <b>MTG2</b> = the 3rd, <b>MTG3</b> = the 4th (every earlier candle went the wrong way). "
        "Doji is never a loss (skipped). B2B = longest run of consecutive losses in hour order: day (restarts every day) / chain (continues across days). "
        "B2B colour: green 0-2, light green 3, yellow 4, orange 5, red 6+.", p9))
    story.append(Spacer(1, 6))

    # ---- overview: best 5 per mode
    story.append(Paragraph("Overview - best 5 per mode (smallest B2B loss)", h2))
    ov = [["Mode"] + ["#%d" % i for i in range(1, 6)]]
    for k in range(kmax + 1):
        row = [MODES[k]]
        for r in ranked(res, k, 5):
            row.append(":%02d %s  B2B %d/%d  WR %.0f%%" % (r["minute"], SIDE[r["side"]], r["b2b_day"], r["b2b_chain"], 100 * r["wr"]))
        ov.append(row + [""] * (6 - len(row)))
    t = Table(ov, colWidths=[56] + [(page[0] - 2 * M - 56) / 5.0] * 5)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                           ("FONTSIZE", (0, 0), (-1, -1), 7.4), ("GRID", (0, 0), (-1, -1), 0.4, LINE),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ZEBRA])]))
    story.append(t)

    # ---- one chart per mode
    head = ["Rank", "Minute", "Trade", "Win", "Loss", "WR %", "B2B day", "B2B chain", "Win breakdown (direct + MTG steps)",
            "Longest loss run - chain (when)"]
    cw = [34, 40, 46, 40, 40, 42, 44, 46, 150, 340]
    for k in range(kmax + 1):
        rows = ranked(res, k, top)
        story.append(PageBreak())
        story.append(Paragraph("%s  -  %s  -  ranked by smallest back-to-back loss" % (pair, MODES[k]), h2))
        data = [head]
        style = [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                 ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, 0), 7.2),
                 ("FONTSIZE", (0, 1), (-1, -1), 7.8), ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                 ("ALIGN", (0, 0), (8, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                 ("GRID", (0, 0), (-1, -1), 0.4, LINE), ("TOPPADDING", (0, 1), (-1, -1), 1.6),
                 ("BOTTOMPADDING", (0, 1), (-1, -1), 1.6)]
        for i, r in enumerate(rows, start=1):
            bd = " + ".join(str(v) for v in r["levels"][:k + 1])
            data.append([r["rank"], ":%02d" % r["minute"], SIDE[r["side"]], r["win"], r["loss"], "%.1f" % (100 * r["wr"]),
                         r["b2b_day"], r["b2b_chain"], bd, Paragraph(r["chain_at"] or "-", pc)])
            if r["rank"] <= 3:
                style.append(("BACKGROUND", (0, i), (2, i), GOLD))
            elif i % 2 == 0:
                style.append(("BACKGROUND", (0, i), (2, i), ZEBRA))
            style.append(("TEXTCOLOR", (2, i), (2, i), GREEN if r["side"] == "G" else RED))
            style.append(("FONTNAME", (2, i), (2, i), "Helvetica-Bold"))
            style.append(("FONTNAME", (5, i), (5, i), "Helvetica-Bold"))
            style.append(("BACKGROUND", (6, i), (6, i), b2b_bg(r["b2b_day"])))
            style.append(("BACKGROUND", (7, i), (7, i), b2b_bg(r["b2b_chain"])))
        t = Table(data, repeatRows=1, colWidths=cw)
        t.setStyle(TableStyle(style))
        story.append(t)

    bio = io.BytesIO()
    doc = SimpleDocTemplate(bio, pagesize=page, leftMargin=M, rightMargin=M, topMargin=M, bottomMargin=M,
                            title="%s M%d ranked minute chart" % (pair, tf))
    doc.build(story)
    bio.seek(0)
    return bio


# ------------------------------------------------------------------ Telegram caption (Bengali)
def bn_text(res, kmax=MAXK, n=3):
    out = []
    for k in range(kmax + 1):
        out.append("%s:" % ["নন-MTG", "MTG1", "MTG2", "MTG3"][k])
        for r in ranked(res, k, n):
            out.append("  %d) :%02d %s • B2B %d/%d • %.0f%%" % (r["rank"], r["minute"], "🟢" if r["side"] == "G" else "🔴",
                                                              r["b2b_day"], r["b2b_chain"], 100 * r["wr"]))
    return "\n".join(out)

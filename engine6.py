"""Engine 6 — Minute Grid, ranked by back-to-back loss (pure numpy + reportlab, no Telegram, no database).

engine.py .. engine4.py are untouched. Engine 6 runs exactly like Engine 3 (same data, same candle / MTG / doji rules,
same pair + minute selection, same B2B day / chain numbers: engine3.analyze is reused 1:1). Only the PDF / CSV layout is new:

 * Engine 3 lists the minutes in time order (:00, :01, :02 ...). Engine 6 does NOT.
 * Charts come in this order:  No-MTG  ->  MTG1  ->  MTG2  ->  MTG3  (only up to the MTG step you choose).
 * Inside every mode there are TWO separate charts: first GREEN (you trade green), then RED (you trade red).
 * In every chart the chosen minutes are sorted by the SMALLEST back-to-back loss inside ONE DAY first
   (sort: B2B day, then B2B chain, then higher win rate, then earlier minute). Rank 1 = the safest minute.
 * Above each chart one line says which minutes have the lowest B2B-day loss.
 * The daily G/R candle tables of Engine 3 are kept at the end (orange header = your selected minute).
"""
import csv
import io

import engine3 as e3

MAXK = e3.MAXK
MODES = e3.MODES
WD = e3.WD
HOURS = e3.HOURS
SIDE = {"G": "GREEN", "R": "RED"}


def analyze(pair, ts, o, c, tf, months, minutes):
    """Same as Engine 3."""
    return e3.analyze(pair, ts, o, c, tf, months, minutes)


def ranked(res, k, side):
    """-> list of row dicts for MTG level k and one side ('G' or 'R'), smallest one-day B2B loss first."""
    rows = []
    for st in res["stats"]:
        x = st["sides"][side]["m"][k]
        rows.append(dict(minute=st["minute"], side=side, g=st["g"], r=st["r"], doji=st["doji"], days=st["days"],
                         win=x["win"], loss=x["loss"], wr=x["wr"], n=x["n"], levels=x["levels"], skipped=x["skipped"],
                         b2b_day=x["b2b_day"], b2b_chain=x["b2b_chain"], day_at=x["b2b_day_at"], chain_at=x["b2b_chain_at"]))
    rows.sort(key=lambda r: (r["b2b_day"], r["b2b_chain"], -r["wr"], r["minute"]))
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    return rows


def best_line(rows, limit=12):
    """'Lowest B2B (one day) = 2  ->  :05 :12 :30' for the minutes that share the smallest value."""
    if not rows:
        return "-"
    lo = rows[0]["b2b_day"]
    ms = [r["minute"] for r in rows if r["b2b_day"] == lo]
    txt = " ".join(":%02d" % m for m in ms[:limit])
    if len(ms) > limit:
        txt += " ... (+%d)" % (len(ms) - limit)
    return "Lowest B2B loss in one day = %d  ->  minutes: %s" % (lo, txt)


# ------------------------------------------------------------------ CSV
def make_csv(res, levels=None):
    """-> (grid_csv bytes, ranked_summary_csv bytes). Grid is the same as Engine 3; the summary is ranked."""
    levels = sorted(set(range(MAXK + 1) if levels is None else levels))
    grid, _ = e3.make_csv(res, max(levels))
    s = io.StringIO()
    w = csv.writer(s)
    w.writerow(["pair", "tf_min", "mode", "trade", "rank", "minute", "days", "green", "red", "doji", "win", "loss",
                "winrate_pct", "win_direct", "win_mtg1", "win_mtg2", "win_mtg3", "step_skipped", "b2b_loss_day",
                "b2b_loss_chain", "b2b_loss_day_when", "b2b_loss_chain_when"])
    for k in levels:
        for side in ("G", "R"):
            for r in ranked(res, k, side):
                lv = r["levels"] + [""] * (MAXK + 1 - len(r["levels"]))
                w.writerow([res["pair"], res["tf"], MODES[k], SIDE[side], r["rank"], "%02d" % r["minute"], r["days"],
                            r["g"], r["r"], r["doji"], r["win"], r["loss"], "%.1f" % (100 * r["wr"]),
                            lv[0], lv[1], lv[2], lv[3], r["skipped"], r["b2b_day"], r["b2b_chain"],
                            r["day_at"], r["chain_at"]])
    return grid, s.getvalue().encode("utf-8-sig")


# ------------------------------------------------------------------ PDF
def make_pdf(res, levels=None):
    levels = sorted(set(range(MAXK + 1) if levels is None else levels))
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Flowable, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    tf, per = res["tf"], 60 // res["tf"]
    sel = set(m // tf for m in res["minutes"])
    pair = res["pair"].replace("_", "/")
    GREEN, RED = colors.Color(0.0, 0.5, 0.25), colors.Color(0.8, 0.15, 0.2)
    NAVY, LIGHT, ZEBRA = colors.Color(0.09, 0.22, 0.4), colors.Color(0.86, 0.91, 0.96), colors.Color(0.94, 0.96, 0.98)
    LINE, GOLD, GREY = colors.Color(0.8, 0.84, 0.9), colors.Color(1.0, 0.96, 0.78), colors.Color(0.45, 0.45, 0.5)
    page = landscape(A4)
    M = 20
    width = page[0] - 2 * M

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

    class DayGrid(Flowable):
        RH, HH, TH, LW = 9.6, 10.5, 12.0, 26.0

        def __init__(self, i):
            super().__init__()
            self.i = i
            self.width = width
            self.height = self.TH + self.HH + len(HOURS) * self.RH + 8

        def draw(self):
            cv = self.canv
            dd = e3.day_date(res["d0"], self.i)
            cv.setFont("Helvetica-Bold", 8)
            cv.setFillColor(NAVY)
            cv.drawString(0, self.height - 9, "%s (%s)   %s   M%d   BDT UTC+6" % (dd.isoformat(), WD[dd.weekday()], pair, tf))
            cw = (self.width - self.LW) / per
            top = self.height - self.TH
            cv.setFillColor(NAVY)
            cv.rect(0, top - self.HH, self.width, self.HH, stroke=0, fill=1)
            cv.setFont("Helvetica-Bold", min(6.5, cw * 0.62))
            cv.setFillColor(colors.white)
            cv.drawCentredString(self.LW / 2, top - self.HH + 3, "H\\M")
            for k in range(per):
                x = self.LW + k * cw
                if k in sel:
                    cv.setFillColor(colors.Color(0.95, 0.6, 0.1))
                    cv.rect(x, top - self.HH, cw, self.HH, stroke=0, fill=1)
                    cv.setFillColor(colors.white)
                cv.drawCentredString(x + cw / 2, top - self.HH + 3, "%02d" % (k * tf))
            fs = min(7.0, cw * 0.66)
            for r, h in enumerate(HOURS):
                y = top - self.HH - (r + 1) * self.RH
                cv.setFillColor(LIGHT)
                cv.rect(0, y, self.LW, self.RH, stroke=0, fill=1)
                if r % 2 == 0:
                    cv.setFillColor(ZEBRA)
                    cv.rect(self.LW, y, self.width - self.LW, self.RH, stroke=0, fill=1)
                for k in sel:
                    cv.setFillColor(colors.Color(1, 0.97, 0.8))
                    cv.rect(self.LW + k * cw, y, cw, self.RH, stroke=0, fill=1)
                cv.setFillColor(NAVY)
                cv.setFont("Helvetica-Bold", 7)
                cv.drawCentredString(self.LW / 2, y + 2.4, "%02d" % h)
                cv.setFont("Helvetica-Bold", fs)
                for k in range(per):
                    n = h * per + k
                    if not res["val"][self.i, n]:
                        continue
                    v = res["dirm"][self.i, n]
                    cv.setFillColor(GREEN if v == 1 else RED if v == -1 else GREY)
                    cv.drawCentredString(self.LW + k * cw + cw / 2, y + 2.4, "G" if v == 1 else "R" if v == -1 else "D")

    h1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=15, textColor=NAVY, spaceAfter=4)
    h2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12, textColor=NAVY, spaceBefore=4, spaceAfter=3)
    hg = ParagraphStyle("hg", fontName="Helvetica-Bold", fontSize=11, textColor=GREEN, spaceBefore=6, spaceAfter=2)
    hr = ParagraphStyle("hr", fontName="Helvetica-Bold", fontSize=11, textColor=RED, spaceBefore=6, spaceAfter=2)
    p9 = ParagraphStyle("p9", fontName="Helvetica", fontSize=8.5, leading=11)
    pb = ParagraphStyle("pb", fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=NAVY, spaceAfter=3)
    pc = ParagraphStyle("pc", fontName="Helvetica", fontSize=6.6, leading=7.8)

    dts = [e3.day_date(res["d0"], i) for i in res["days"]]
    story = [Paragraph("%s  -  Engine 6 Ranked Minute Charts  (M%d)" % (pair, tf), h1)]
    story.append(Paragraph(
        "Period: %s to %s  (%d trading days, last %s)   |   Entry hours 12:00-23:59 BDT (UTC+6)   |   Entry minutes: %s" % (
            dts[0], dts[-1], len(dts), e3.period_label_en(res["months"]),
            ", ".join(":%02d" % m for m in res["minutes"])), p9))
    story.append(Paragraph(
        "<b>How to read:</b> the charts follow this order: <b>No-MTG</b>, then <b>MTG1</b>, <b>MTG2</b>, <b>MTG3</b> (up to the step you chose). "
        "Every mode has a <b>GREEN chart</b> (you trade green at that minute every hour) and then a <b>RED chart</b> (you trade red). "
        "Inside a chart the minutes are NOT in time order: they are <b>ranked by the smallest back-to-back (B2B) loss inside one day first</b> "
        "(then B2B chain, then higher win rate). Rank 1 = the safest minute. "
        "<b>MTG1</b> = the 2nd candle may win, <b>MTG2</b> = the 3rd, <b>MTG3</b> = the 4th (every earlier candle went the wrong way). "
        "Doji is never a loss (skipped). B2B = longest run of consecutive losses in hour order: day (restarts every day) / chain (continues across days). "
        "B2B colour: green 0-2, light green 3, yellow 4, orange 5, red 6+. Daily candle tables at the end: orange header = your selected minute(s).", p9))

    head = ["Rank", "Minute", "Win", "Loss", "WR %", "B2B day", "B2B chain", "Win breakdown (direct + MTG steps)",
            "Longest loss run - day (when)", "Longest loss run - chain (when)"]
    cw = [32, 40, 38, 38, 40, 44, 46, 130, 200, 214]

    def chart(k, side):
        rows = ranked(res, k, side)
        data = [head]
        style = [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                 ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, 0), 7.2),
                 ("FONTSIZE", (0, 1), (-1, -1), 7.8), ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                 ("ALIGN", (0, 0), (7, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                 ("GRID", (0, 0), (-1, -1), 0.4, LINE), ("TOPPADDING", (0, 1), (-1, -1), 1.6),
                 ("BOTTOMPADDING", (0, 1), (-1, -1), 1.6)]
        for i, r in enumerate(rows, start=1):
            bd = " + ".join(str(v) for v in r["levels"][:k + 1])
            data.append([r["rank"], ":%02d" % r["minute"], r["win"], r["loss"], "%.1f" % (100 * r["wr"]),
                         r["b2b_day"], r["b2b_chain"], bd, Paragraph(r["day_at"] or "-", pc), Paragraph(r["chain_at"] or "-", pc)])
            if r["rank"] <= 3:
                style.append(("BACKGROUND", (0, i), (1, i), GOLD))
            elif i % 2 == 0:
                style.append(("BACKGROUND", (0, i), (1, i), ZEBRA))
            style.append(("FONTNAME", (1, i), (1, i), "Helvetica-Bold"))
            style.append(("FONTNAME", (4, i), (4, i), "Helvetica-Bold"))
            style.append(("BACKGROUND", (5, i), (5, i), b2b_bg(r["b2b_day"])))
            style.append(("BACKGROUND", (6, i), (6, i), b2b_bg(r["b2b_chain"])))
        t = Table(data, repeatRows=1, colWidths=cw)
        t.setStyle(TableStyle(style))
        title = Paragraph("%s  -  %s" % (MODES[k], "GREEN chart (trade green)" if side == "G" else "RED chart (trade red)"),
                          hg if side == "G" else hr)
        return [KeepTogether([title, Paragraph(best_line(rows), pb)]), t]

    for k in levels:
        story.append(PageBreak())
        story.append(Paragraph("%s  -  %s  -  ranked by smallest back-to-back loss in one day" % (pair, MODES[k]), h2))
        story.extend(chart(k, "G"))
        story.append(Spacer(1, 8))
        story.extend(chart(k, "R"))

    story.append(PageBreak())
    story.append(Paragraph("Daily candle tables (G = green, R = red, D = doji)", h2))
    for i in res["days"]:
        story.append(KeepTogether([DayGrid(i)]))
    bio = io.BytesIO()
    doc = SimpleDocTemplate(bio, pagesize=page, leftMargin=M, rightMargin=M, topMargin=M, bottomMargin=M,
                            title="%s M%d ranked minute charts" % (pair, tf))
    doc.build(story)
    bio.seek(0)
    return bio

"""Engine 10 report files (PDF + CSV) built from the radar build result. Pure reportlab / csv, no Telegram, no database.
PDF text is English (Bengali text needs a shaping library that most servers do not have); the Telegram text stays Bengali."""
import csv
import io

import engine3 as e3
import engine9
import pairs

SIDE = {"G": "GREEN (UP)", "R": "RED (DOWN)"}
MODES = e3.MODES


def _rows(per, st):
    """Flat list of dicts: one per (pair, MTG, side, minute) candidate that Engine 9 produced."""
    out = []
    for e in per:
        if e is None:
            continue
        res = e["res"]
        for k in res["ks"]:
            for side in ("G", "R"):
                d = res["per"][k][side]
                for r in d["rows"]:
                    if r["sig"] == 0:
                        why = "no signals"
                    elif r["win"] != r["sig"]:
                        why = "win rate below 100%"
                    elif r["sig"] < st["minsig"]:
                        why = "fewer than %d signals" % st["minsig"]
                    else:
                        why = "ok"
                    out.append(dict(pair=e["pair"], res=res, k=k, side=side, lo=d["lo"], minute=r["minute"], sig=r["sig"], win=r["win"],
                                    loss=r["loss"], doji=r["doji"], signals=r["signals"], why=why, radar=(why == "ok")))
    return out


def _on_radar(rows):
    """A (pair, side, minute) is on the radar once, with its lowest qualifying MTG (same rule as radar.rules_from)."""
    seen, res = set(), []
    for x in sorted(rows, key=lambda x: (x["k"],)):
        key = (x["pair"], x["side"], x["minute"])
        if x["radar"] and key not in seen:
            seen.add(key)
            res.append(x)
        elif x["radar"]:
            x["why"] = "ok (higher MTG, a lower MTG already on radar)"
            x["radar"] = False
    return res


def make_csv(per, st):
    """-> (slots_csv bytes, signals_csv bytes)."""
    rows = _rows(per, st)
    _on_radar(rows)
    s = io.StringIO()
    w = csv.writer(s)
    w.writerow(["pair", "tf_min", "months", "mtg", "b2b_loss_n", "rule", "side", "minute", "lowest_b2b_day_loss", "signals", "win",
                "of_which_doji_win", "loss", "winrate_pct", "on_radar", "note"])
    for x in sorted(rows, key=lambda x: (x["pair"], x["side"] != "G", x["minute"], x["k"])):
        w.writerow([x["pair"], st["tf"], st["months_label"], MODES[x["k"]], st["n"], ("atleast_" if st["mode"] == "atleast" else "exact_") + "day",
                    SIDE[x["side"]], "%02d" % x["minute"], x["lo"], x["sig"], x["win"], x["doji"], x["loss"],
                    ("%.1f" % (100.0 * x["win"] / x["sig"])) if x["sig"] else "", "YES" if x["radar"] else "NO", x["why"]])
    g = io.StringIO()
    w = csv.writer(g)
    w.writerow(["pair", "side", "minute", "mtg", "date", "weekday", "hour", "result", "win_at", "doji_win", "on_radar"])
    for x in sorted(rows, key=lambda x: (x["pair"], x["side"] != "G", x["minute"], x["k"])):
        for dayno, h, v, lvl, dj in x["signals"]:
            dd = e3.day_date(x["res"]["d0"], x["res"]["days"][dayno])
            w.writerow([x["pair"], SIDE[x["side"]], "%02d" % x["minute"], MODES[x["k"]], dd.isoformat(), e3.WD[dd.weekday()], "%02d" % (x["res"]["hours"][h]),
                        "WIN" if v == 1 else "LOSS", (("direct" if lvl == 0 else "MTG%d" % lvl) if v == 1 else ""), "YES" if dj else "",
                        "YES" if x["radar"] else "NO"])
    return s.getvalue().encode("utf-8-sig"), g.getvalue().encode("utf-8-sig")


def make_pdf(per, st):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    rows = _rows(per, st)
    onr = _on_radar(rows)
    NAVY, ZEBRA, LINE = colors.Color(0.09, 0.22, 0.4), colors.Color(0.94, 0.96, 0.98), colors.Color(0.8, 0.84, 0.9)
    GREEN, RED, GREY = colors.Color(0.0, 0.5, 0.25), colors.Color(0.8, 0.15, 0.2), colors.Color(0.45, 0.45, 0.45)
    h1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=16, textColor=NAVY, spaceAfter=4)
    h2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11, textColor=NAVY, spaceBefore=8, spaceAfter=4)
    p9 = ParagraphStyle("p9", fontName="Helvetica", fontSize=8.5, leading=11)
    buf = io.BytesIO()
    page = landscape(A4)
    doc = SimpleDocTemplate(buf, pagesize=page, leftMargin=24, rightMargin=24, topMargin=22, bottomMargin=22,
                            title="Signal Radar (Engine 10)")

    def table(head, body, widths, colour_col=None):
        t = Table([head] + body, repeatRows=1, colWidths=widths)
        style = [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                 ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8),
                 ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("GRID", (0, 0), (-1, -1), 0.4, LINE)]
        for i in range(1, len(body) + 1):
            if i % 2 == 0:
                style.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
            if colour_col is not None:
                v = str(body[i - 1][colour_col])
                style.append(("TEXTCOLOR", (colour_col, i), (colour_col, i), GREEN if v.startswith("GREEN") else RED))
                style.append(("FONTNAME", (colour_col, i), (colour_col, i), "Helvetica-Bold"))
        t.setStyle(TableStyle(style))
        return t

    days_all = [len(e["res"]["days"]) for e in per if e]
    dts = [e3.day_date(e["res"]["d0"], e["res"]["days"][0]) for e in per if e] + [e3.day_date(e["res"]["d0"], e["res"]["days"][-1]) for e in per if e]
    story = [Paragraph("Signal Radar - Engine 10 (back-test details)", h1)]
    story.append(Paragraph(
        "Time frame: M%d | Period: %s (%s to %s, up to %d trading days) | MTG: %s | Rule: %d back-to-back losses (%s, day scope) | "
        "Radar rule: win rate 100%% with at least %d signals | Entry hours %s BDT (UTC+6) | Pairs: %s" % (
            st["tf"], st["months_label"], min(dts) if dts else "-", max(dts) if dts else "-", max(days_all) if days_all else 0,
            ", ".join(MODES[k] for k in st["mtg"]), st["n"], "N or more" if st["mode"] == "atleast" else "exactly N", st["minsig"],
            ", ".join(sorted({e3.hours_label(e["res"]["hours"]) for e in per if e})) or "-",
            ", ".join(pairs.pretty(p) for p in st["pairs"])), p9))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "<b>How to read:</b> Engine 8 finds, for each pair and colour, the minutes with the lowest one-day back-to-back loss. For exactly those minutes "
        "Engine 4 counts the <b>signals</b> (the trade right after the chosen number of back-to-back losses in the same day). "
        "A doji candle (open = close) counts as a WIN. A minute goes on the radar only when every signal won and there are enough signals. "
        "In live mode the radar sends the alert about 45 seconds before the slot when today's earlier hours show the same losing streak.", p9))

    # ---- coverage
    allg = sorted({x["minute"] for x in onr if x["side"] == "G"})
    allr = sorted({x["minute"] for x in onr if x["side"] == "R"})
    story.append(Paragraph("Radar minutes of all pairs together (each minute once)", h2))
    story.append(Paragraph("<font color='#00803f'><b>GREEN / UP (%d):</b></font> %s" % (len(allg), " ".join(":%02d" % m for m in allg) or "-"), p9))
    story.append(Paragraph("<font color='#cc2633'><b>RED / DOWN (%d):</b></font> %s" % (len(allr), " ".join(":%02d" % m for m in allr) or "-"), p9))

    head = ["Pair", "Trade", "Minute", "MTG", "Lowest B2B loss", "Signals", "Win", "of which doji", "Loss", "WR %"]
    W = [70, 80, 50, 50, 85, 55, 45, 75, 45, 50]

    def body(items):
        return [[pairs.pretty(x["pair"]), SIDE[x["side"]], ":%02d" % x["minute"], MODES[x["k"]], x["lo"], x["sig"], x["win"], x["doji"], x["loss"],
                 ("%.1f" % (100.0 * x["win"] / x["sig"])) if x["sig"] else "-"] for x in items]

    key = lambda x: (pairs.PAIRS.index(x["pair"]) if x["pair"] in pairs.PAIRS else 99, x["side"] != "G", x["minute"], x["k"])  # noqa: E731
    story.append(Paragraph("Radar slots (%d)" % len(onr), h2))
    story.append(table(head, body(sorted(onr, key=key)), W, 1) if onr else Paragraph("No slot qualified.", p9))
    rest = sorted([x for x in rows if not x["radar"]], key=key)
    if rest:
        story.append(Paragraph("Candidate minutes NOT on the radar (%d) - nothing is hidden, this is why they were left out" % len(rest), h2))
        story.append(table(head + ["Reason"], [b + [x["why"]] for b, x in zip(body(rest), rest)], W + [140], 1))

    # ---- signal details
    for x in sorted(onr, key=key):
        story.append(PageBreak())
        story.append(Paragraph("<b>%s - %s :%02d - %s</b> (%d signals: %d win, %d loss, %d of the wins are doji)" % (
            pairs.pretty(x["pair"]), SIDE[x["side"]], x["minute"], MODES[x["k"]], x["sig"], x["win"], x["loss"], x["doji"]), h2))
        det = []
        for dayno, h, v, lvl, dj in x["signals"]:
            dd = e3.day_date(x["res"]["d0"], x["res"]["days"][dayno])
            det.append([dd.isoformat(), e3.WD[dd.weekday()], "%02d:%02d" % (x["res"]["hours"][h], x["minute"]), "WIN" if v == 1 else "LOSS",
                        (("direct" if lvl == 0 else "MTG%d" % lvl) if v == 1 else "-"), "doji" if dj else ""])
        story.append(table(["Date", "Day", "Entry time (BDT)", "Result", "Win at", "Doji"], det, [90, 50, 110, 60, 70, 50]))
    doc.build(story)
    buf.seek(0)
    return buf

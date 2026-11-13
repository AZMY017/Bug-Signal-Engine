"""Engine 3 — Minute Grid (pure numpy + reportlab, no Telegram, no database).

Separate from engine.py / engine2.py (both untouched).

What it does (one pair, one timeframe, last N days, chosen minute slots):
 * Candle colour per tf-candle: close > open = G, close < open = R, open == close = D (doji). Time = UTC+6 (BDT), hours 12..23.
 * For a chosen minute (e.g. :00) the entry candle is that minute of every hour (12:00, 13:00 ... 23:00) of every day.
   Win for a side (G = trade green, R = trade red):
     No-MTG : entry candle goes that way.
     MTG1   : entry candle goes that way (direct win) OR the entry candle goes the other way and the NEXT candle goes
              that way (MTG win). Entry wrong + next wrong = loss.
 * Doji is shown as D, is never a loss: an entry that is a doji, or an MTG step that lands on a doji / missing candle,
   is skipped (counted separately, neither win nor loss, does not break a loss streak).
 * Back-to-back loss = consecutive losses in hour order (12, 13 ... 23). Reported two ways:
   "day"   = streak restarts every day,   "chain" = streak continues from one day into the next.
"""
import csv
import datetime as dt
import io

import numpy as np

OFF = 6 * 3600                 # UTC+6
HOURS = list(range(12, 24))    # 12:00 .. 23:59 BDT
PERIOD_DAYS = {1: 30, 2: 60, 3: 90, 6: 180, 12: 365}
WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ------------------------------------------------------------------ candles
def to_tf(ts, o, c, tf):
    """M1 rows (ts unix sec, open, close) -> (local start seconds int64, direction int8) of tf-minute candles.
    A tf>1 candle needs all its tf one-minute candles (otherwise it is missing, never guessed)."""
    loc = np.asarray(ts, np.int64) + OFF
    o = np.asarray(o, np.float64)
    c = np.asarray(c, np.float64)
    if tf == 1:
        return loc - loc % 60, np.sign(c - o).astype(np.int8)
    b = loc // (60 * tf)
    chg = np.flatnonzero(np.diff(b)) + 1
    first = np.concatenate(([0], chg))
    last = np.concatenate((chg - 1, [len(b) - 1]))
    ok = ((last - first + 1) == tf) & ((loc[last] - loc[first]) == (tf - 1) * 60)
    first, last = first[ok], last[ok]
    return b[first] * 60 * tf, np.sign(c[last] - o[first]).astype(np.int8)


def build_grid(lt, d, tf):
    """-> (first_day, dirm (D,S) int8, val (D,S) bool); S = candles per day. Day number = local day since epoch."""
    S = 1440 // tf
    day = lt // 86400
    d0 = int(day.min())
    nd = int(day.max()) - d0 + 1
    dirm = np.zeros((nd, S), np.int8)
    val = np.zeros((nd, S), bool)
    r = day - d0
    sl = (lt % 86400) // (60 * tf)
    dirm[r, sl] = d
    val[r, sl] = True
    return d0, dirm, val


def pick_days(dirm, val, tf, months):
    """Indices of the last N calendar days that have candles in 12..23 (the latest day is dropped if still running)."""
    per = 60 // tf
    win = val[:, HOURS[0] * per:(HOURS[-1] + 1) * per]
    has = win.any(1)
    if not has.any():
        return np.zeros(0, np.int64)
    last = int(np.flatnonzero(has)[-1])
    if not val[last, HOURS[-1] * per:(HOURS[-1] + 1) * per].any():     # hour 23 empty -> the day is not finished yet
        has[last] = False
        if not has.any():
            return np.zeros(0, np.int64)
        last = int(np.flatnonzero(has)[-1])
    first = max(0, last - PERIOD_DAYS[months] + 1)
    idx = np.arange(first, last + 1)
    return idx[has[idx]]


# ------------------------------------------------------------------ statistics
def _fmt_run(d0, days, di0, h0, di1, h1):
    """Readable place of a loss streak: same day -> '2026-10-21 14h-19h', across days -> '2026-10-21 22h -> 2026-10-22 03h'."""
    a, b = day_date(d0, days[di0]).isoformat(), day_date(d0, days[di1]).isoformat()
    if di0 == di1:
        return "%s %02dh" % (a, h0) if h0 == h1 else "%s %02dh-%02dh" % (a, h0, h1)
    return "%s %02dh -> %s %02dh" % (a, h0, b, h1)


def _streaks(out, days=None, d0=0):
    """out (days, hours) int8: 1 win, -1 loss, 0 skipped (neither breaks nor extends a streak).
    -> (max streak inside a day, where, max streak chained over days, where). 'where' = first time it happened."""
    nd, k = out.shape
    if days is None:
        days = np.arange(nd)
    best_d, where_d = 0, ""
    best_c, where_c = 0, ""
    cc, c_start = 0, None
    for i in range(nd):
        dc, d_start = 0, None
        for j in range(k):
            v = out[i, j]
            if v == -1:
                if dc == 0:
                    d_start = j
                dc += 1
                if dc > best_d:
                    best_d, where_d = dc, _fmt_run(d0, days, i, HOURS[d_start], i, HOURS[j])
                if cc == 0:
                    c_start = (i, j)
                cc += 1
                if cc > best_c:
                    best_c, where_c = cc, _fmt_run(d0, days, c_start[0], HOURS[c_start[1]], i, HOURS[j])
            elif v == 1:
                dc = 0
                cc = 0
    return best_d, where_d, best_c, where_c


def _pack(out, direct, mtg_w, days=None, d0=0):
    w = int((out == 1).sum())
    lo = int((out == -1).sum())
    n = w + lo
    bd, wd, bc, wc = _streaks(out, days, d0)
    return dict(win=w, loss=lo, n=n, wr=(w / n if n else 0.0), direct=direct, mtg=mtg_w,
                b2b_day=bd, b2b_day_at=wd, b2b_chain=bc, b2b_chain_at=wc)


def minute_stats(dirm, val, tf, minute, days, d0=0):
    """Statistics of the entry slot `minute` (0..59, multiple of tf) over the chosen days."""
    per = 60 // tf
    S = 1440 // tf
    j = minute // tf
    fd = np.concatenate((dirm.reshape(-1), [0])).astype(np.int8)
    fv = np.concatenate((val.reshape(-1), [False]))
    idx = days[:, None] * S + np.array([h * per + j for h in HOURS])[None, :]
    e, ev, nx, nv = fd[idx], fv[idx], fd[idx + 1], fv[idx + 1]
    g = int((ev & (e == 1)).sum())
    r = int((ev & (e == -1)).sum())
    dj = int((ev & (e == 0)).sum())
    res = dict(minute=minute, days=len(days), g=g, r=r, doji=dj, sides={})
    for name, s in (("G", 1), ("R", -1)):
        non = np.zeros(e.shape, np.int8)
        non[ev & (e == s)] = 1
        non[ev & (e == -s)] = -1
        direct = ev & (e == s)
        mwin = ev & (e == -s) & nv & (nx == s)
        mloss = ev & (e == -s) & nv & (nx == -s)
        mt = np.zeros(e.shape, np.int8)
        mt[direct | mwin] = 1
        mt[mloss] = -1
        skipped = int((ev & (e == -s) & ~(mwin | mloss)).sum())
        res["sides"][name] = dict(non=_pack(non, int(direct.sum()), 0, days, d0),
                                  mtg=_pack(mt, int(direct.sum()), int(mwin.sum()), days, d0), mtg_skipped=skipped)
    res["dominant"] = "G" if g >= r else "R"
    return res


def analyze(pair, ts, o, c, tf, months, minutes):
    """-> dict with everything the PDF/CSV/text need, or None when there is not enough data."""
    if len(ts) < 500:
        return None
    lt, d = to_tf(ts, o, c, tf)
    if len(lt) == 0:
        return None
    d0, dirm, val = build_grid(lt, d, tf)
    days = pick_days(dirm, val, tf, months)
    if len(days) == 0:
        return None
    return dict(pair=pair, tf=tf, months=months, d0=d0, dirm=dirm, val=val, days=days,
                minutes=sorted(minutes), stats=[minute_stats(dirm, val, tf, m, days, d0) for m in minutes_sorted(minutes)])


def minutes_sorted(minutes):
    return sorted(minutes)


def day_date(d0, i):
    return dt.date(1970, 1, 1) + dt.timedelta(days=int(d0 + i))


# ------------------------------------------------------------------ CSV
def _letter(dv, vv):
    return ("G" if dv == 1 else "R" if dv == -1 else "D") if vv else ""


def make_csv(res):
    """-> (grid_csv bytes, summary_csv bytes). Grid: one row per day+hour, one column per minute slot."""
    tf, per, S = res["tf"], 60 // res["tf"], 1440 // res["tf"]
    g = io.StringIO()
    w = csv.writer(g)
    w.writerow(["date", "weekday", "hour"] + ["%02d" % (k * tf) for k in range(per)])
    for i in res["days"]:
        dd = day_date(res["d0"], i)
        for h in HOURS:
            w.writerow([dd.isoformat(), WD[dd.weekday()], "%02d" % h] +
                       [_letter(res["dirm"][i, h * per + k], res["val"][i, h * per + k]) for k in range(per)])
    s = io.StringIO()
    w = csv.writer(s)
    w.writerow(["pair", "tf_min", "minute", "days", "green", "red", "doji", "side", "mode", "win", "loss", "winrate_pct",
                "direct_win", "mtg_win", "mtg_step_skipped", "b2b_loss_day", "b2b_loss_day_when", "b2b_loss_chain", "b2b_loss_chain_when"])
    for st in res["stats"]:
        for side in ("G", "R"):
            sd = st["sides"][side]
            for mode, key in (("No-MTG", "non"), ("MTG1", "mtg")):
                x = sd[key]
                w.writerow([res["pair"], tf, "%02d" % st["minute"], st["days"], st["g"], st["r"], st["doji"],
                            "GREEN" if side == "G" else "RED", mode, x["win"], x["loss"], "%.1f" % (100 * x["wr"]),
                            x["direct"] if key == "mtg" else "", x["mtg"] if key == "mtg" else "",
                            sd["mtg_skipped"] if key == "mtg" else "", x["b2b_day"], x["b2b_day_at"], x["b2b_chain"], x["b2b_chain_at"]])
    return g.getvalue().encode("utf-8-sig"), s.getvalue().encode("utf-8-sig")


# ------------------------------------------------------------------ PDF
def make_pdf(res):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Flowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    tf, per = res["tf"], 60 // res["tf"]
    sel = set(m // tf for m in res["minutes"])
    pair = res["pair"].replace("_", "/")
    GREEN, RED, GREY = colors.Color(0.0, 0.5, 0.25), colors.Color(0.8, 0.15, 0.2), colors.Color(0.45, 0.45, 0.5)
    NAVY, LIGHT, ZEBRA, SELC = colors.Color(0.09, 0.22, 0.4), colors.Color(0.86, 0.91, 0.96), colors.Color(0.94, 0.96, 0.98), colors.Color(1, 0.93, 0.6)
    page = landscape(A4)
    M = 20
    width = page[0] - 2 * M

    class DayGrid(Flowable):
        RH, HH, TH, LW = 9.6, 10.5, 12.0, 26.0

        def __init__(self, i):
            super().__init__()
            self.i = i
            self.width = width
            self.height = self.TH + self.HH + len(HOURS) * self.RH + 8

        def draw(self):
            cv = self.canv
            dd = day_date(res["d0"], self.i)
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
    p9 = ParagraphStyle("p9", fontName="Helvetica", fontSize=8.5, leading=11)
    pc = ParagraphStyle("pc", fontName="Helvetica", fontSize=6.3, leading=7.4, alignment=1)

    def b2b_cell(x):
        return Paragraph("<b>%d / %d</b><br/>day: %s<br/>chain: %s" % (
            x["b2b_day"], x["b2b_chain"], x["b2b_day_at"] or "-", x["b2b_chain_at"] or "-"), pc)
    story = [Paragraph("%s  -  Engine 3 Minute Grid  (M%d)" % (pair, tf), h1)]
    dts = [day_date(res["d0"], i) for i in res["days"]]
    story.append(Paragraph(
        "Period: %s to %s  (%d trading days, last %s)   |   Entry hours 12:00-23:%02d BDT (UTC+6)   |   Entry minutes: %s" % (
            dts[0], dts[-1], len(dts), {1: "1 month", 2: "2 months", 3: "3 months", 6: "6 months", 12: "1 year"}[res["months"]],
            59, ", ".join(":%02d" % m for m in res["minutes"])), p9))
    story.append(Spacer(1, 6))
    head = ["Min", "Days", "G", "R", "Doji", "Trade", "NoMTG\nWin", "NoMTG\nLoss", "NoMTG\nWR%", "NoMTG\nB2B day/chain",
            "MTG1\nWin", "(direct\n+ MTG)", "MTG1\nLoss", "MTG1\nWR%", "MTG1\nB2B day/chain"]
    rows = [head]
    style = [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
             ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, 0), 7),
             ("FONTSIZE", (0, 1), (-1, -1), 8), ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
             ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("GRID", (0, 0), (-1, -1), 0.4, colors.Color(0.8, 0.84, 0.9))]
    for st in res["stats"]:
        for side in ("G", "R"):
            sd = st["sides"][side]
            a, b = sd["non"], sd["mtg"]
            r = len(rows)
            first = side == "G"
            rows.append([":%02d" % st["minute"] if first else "", st["days"] if first else "", st["g"] if first else "",
                         st["r"] if first else "", st["doji"] if first else "",
                         "GREEN" if side == "G" else "RED", a["win"], a["loss"], "%.1f" % (100 * a["wr"]),
                         b2b_cell(a), b["win"], "%d + %d" % (b["direct"], b["mtg"]),
                         b["loss"], "%.1f" % (100 * b["wr"]), b2b_cell(b)])
            style.append(("TEXTCOLOR", (5, r), (5, r), GREEN if side == "G" else RED))
            style.append(("FONTNAME", (5, r), (5, r), "Helvetica-Bold"))
            if side == st["dominant"]:
                style.append(("BACKGROUND", (0, r), (-1, r), colors.Color(0.93, 0.97, 0.9)))
            if first:
                style.append(("LINEABOVE", (0, r), (-1, r), 0.9, NAVY))
    t = Table(rows, repeatRows=1, colWidths=[28, 28, 28, 28, 28, 40, 36, 36, 40, 112, 36, 50, 36, 40, 112])
    t.setStyle(TableStyle(style))
    story.append(t)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>How to read:</b> Green rows = trade GREEN at that minute, red rows = trade RED. Light-green row = the colour that appears more often at that minute. "
        "No-MTG win = the entry candle goes your way. MTG1 win = entry goes your way, OR entry goes the other way and the next candle goes your way. "
        "Doji (open = close) is shown as D and is never a loss: a doji entry, or an MTG step landing on a doji, is skipped (not counted). "
        "B2B = longest run of back-to-back losses in hour order: 'day' (restarts every day) / 'chain' (continues across days); under the numbers: WHEN it happened (first time, BDT date and hour). "
        "Daily tables below: orange header = your selected minute(s).", p9))
    story.append(Spacer(1, 8))
    for n, i in enumerate(res["days"]):
        story.append(KeepTogether([DayGrid(i)]))
    bio = io.BytesIO()
    doc = SimpleDocTemplate(bio, pagesize=page, leftMargin=M, rightMargin=M, topMargin=M, bottomMargin=M,
                            title="%s M%d minute grid" % (pair, tf))
    doc.build(story)
    bio.seek(0)
    return bio


# ------------------------------------------------------------------ Telegram text (Bengali)
def bn_text(res, limit=8):
    out = []
    for st in res["stats"][:limit]:
        tot = st["g"] + st["r"]
        gp = round(100 * st["g"] / tot) if tot else 0
        out.append(":%02d  (%d দিন • গ্রিন %d%% • রেড %d%% • ডোজি %d)" % (st["minute"], st["days"], gp, 100 - gp if tot else 0, st["doji"]))
        for side, icon, nm in (("G", "🟢", "গ্রিন"), ("R", "🔴", "রেড")):
            sd = st["sides"][side]
            a, b = sd["non"], sd["mtg"]
            star = " ⭐" if side == st["dominant"] else ""
            out.append("%s %s%s: নন-MTG %.0f%% (%d/%d) • MTG1 %.0f%% (%d/%d) • B2B লস %d/%d | %d/%d" % (
                icon, nm, star, 100 * a["wr"], a["win"], a["n"], 100 * b["wr"], b["win"], b["n"],
                a["b2b_day"], a["b2b_chain"], b["b2b_day"], b["b2b_chain"]))
            out.append("    📅 সর্বোচ্চ লস কবে — নন-MTG: দিনে %s; চেইন %s" % (a["b2b_day_at"] or "-", a["b2b_chain_at"] or "-"))
            out.append("    📅 MTG1: দিনে %s; চেইন %s" % (b["b2b_day_at"] or "-", b["b2b_chain_at"] or "-"))
        out.append("")
    if len(res["stats"]) > limit:
        out.append("… বাকি মিনিটগুলো PDF/CSV-তে।")
    return "\n".join(out)

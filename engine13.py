"""Engine 13 — Engine 12 (automatic back-test + replay) + EMA trend filter (same rules as /filter of the OTC bot), in ONE command.

Own file, fully self-contained (it imports nothing from engine12.py, radar11.py, otctest.py ...; deleting any of them never breaks it).
For every test day D: the radar list is built ONLY from candles before D, replayed on D (like Engine 12), then every signal is checked
against a grid of Fast/Slow EMAs (price above both = UP trend, below both = DOWN, between = no trend). The best EMA combo of day D is
chosen with the signals of the EARLIER test days only (walk-forward), so the result is an honest out-of-sample test.
The embedded engine3/4/6/8/9 below are the same identical copies that Engine 12 carries."""
import types


# ==========================================================================
# EMBEDDED COPY of engine3.py (identical code, private to this file)
# ==========================================================================
def _build_engine3():
    def _g():
        return _ns

    """Engine 3 — Minute Grid (pure numpy + reportlab, no Telegram, no database).

    Separate from engine.py / engine2.py (both untouched).

    What it does (one pair, one timeframe, last N days, chosen minute slots):
     * Candle colour per tf-candle: close > open = G, close < open = R, open == close = D (doji). Time = UTC+6 (BDT), hours 12..23.
     * For a chosen minute (e.g. :00) the entry candle is that minute of every hour (12:00, 13:00 ... 23:00) of every day.
       Win for a side (G = trade green, R = trade red):
         No-MTG : entry candle goes that way.
         MTG k  : entry goes that way (direct win), OR the entry and the next candles go the other way and candle k+1 (k = 1..3)
                  goes that way: MTG1 = 2nd candle wins, MTG2 = 3rd candle, MTG3 = 4th candle. All k+1 candles wrong = loss.
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
    HOURS = list(range(12, 24))    # real market: entries 12:00 .. 23:59 BDT
    ALL_HOURS = list(range(24))    # OTC market: trades round the clock, entries 00:00 .. 23:59 BDT


    def hours_for(pair):
        """Entry hours (BDT) of a pair: real market 12..23, OTC pair (name ends with _otc) 0..23."""
        return ALL_HOURS if pair.endswith("_otc") else HOURS


    def hours_label(hours):
        return "%02d:00-23:59" % hours[0]
    PERIOD_DAYS = {1: 30, 2: 60, 3: 90, 6: 180, 12: 365}


    def period_days(months):
        """Days for a period given in months (any number, e.g. 4, 5 or 1.5 for 45 days)."""
        return PERIOD_DAYS.get(months) or max(1, int(round(months * 30)))


    def period_label_en(months):
        d = period_days(months)
        if d == 365:
            return "1 year"
        if d % 30 == 0 and d != 360:
            m = d // 30
            return "1 month" if m == 1 else "%d months" % m
        return "%d days" % d
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


    def pick_days(dirm, val, tf, months, hours=None):
        """Indices of the last N calendar days that have candles in the entry hours (the latest day is dropped if still running)."""
        HOURS = hours or _g()["HOURS"]
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
        first = max(0, last - period_days(months) + 1)
        idx = np.arange(first, last + 1)
        return idx[has[idx]]


    # ------------------------------------------------------------------ statistics
    def _fmt_run(d0, days, di0, h0, di1, h1):
        """Readable place of a loss streak: same day -> '2026-10-21 14h-19h', across days -> '2026-10-21 22h -> 2026-10-22 03h'."""
        a, b = day_date(d0, days[di0]).isoformat(), day_date(d0, days[di1]).isoformat()
        if di0 == di1:
            return "%s %02dh" % (a, h0) if h0 == h1 else "%s %02dh-%02dh" % (a, h0, h1)
        return "%s %02dh -> %s %02dh" % (a, h0, b, h1)


    def _streaks(out, days=None, d0=0, hours=None):
        """out (days, hours) int8: 1 win, -1 loss, 0 skipped (neither breaks nor extends a streak).
        -> (max streak inside a day, where, max streak chained over days, where). 'where' = first time it happened."""
        nd, k = out.shape
        H = hours or HOURS
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
                        best_d, where_d = dc, _fmt_run(d0, days, i, H[d_start], i, H[j])
                    if cc == 0:
                        c_start = (i, j)
                    cc += 1
                    if cc > best_c:
                        best_c, where_c = cc, _fmt_run(d0, days, c_start[0], H[c_start[1]], i, H[j])
                elif v == 1:
                    dc = 0
                    cc = 0
        return best_d, where_d, best_c, where_c


    MAXK = 3
    MODES = ["No-MTG", "MTG1", "MTG2", "MTG3"]


    def _pack(out, levels, skipped, days=None, d0=0, hours=None):
        w = int((out == 1).sum())
        lo = int((out == -1).sum())
        n = w + lo
        bd, wd, bc, wc = _streaks(out, days, d0, hours)
        return dict(win=w, loss=lo, n=n, wr=(w / n if n else 0.0), levels=levels, skipped=skipped,
                    b2b_day=bd, b2b_day_at=wd, b2b_chain=bc, b2b_chain_at=wc)


    def minute_stats(dirm, val, tf, minute, days, d0=0, hours=None):
        """Statistics of the entry slot `minute` (0..59, multiple of tf) over the chosen days, for No-MTG and MTG1..MTG3.
        sides[G|R]['m'][k] = pack for MTG k (k = 0 means no MTG)."""
        per = 60 // tf
        S = 1440 // tf
        j = minute // tf
        HOURS = hours or _g()["HOURS"]
        fd = np.concatenate((dirm.reshape(-1), np.zeros(MAXK, np.int8))).astype(np.int8)
        fv = np.concatenate((val.reshape(-1), np.zeros(MAXK, bool)))
        idx = days[:, None] * S + np.array([h * per + j for h in HOURS])[None, :]
        e, ev = fd[idx], fv[idx]
        g = int((ev & (e == 1)).sum())
        r = int((ev & (e == -1)).sum())
        dj = int((ev & (e == 0)).sum())
        res = dict(minute=minute, days=len(days), g=g, r=r, doji=dj, sides={})
        for name, s in (("G", 1), ("R", -1)):
            direct = ev & (e == s)
            alive = ev & (e == -s)                 # entry wrong, still undecided
            win_lv = [direct]                      # win at level 0..k
            alive_after = [alive]                  # still wrong after step k  (-> loss for MTG k)
            skip_at = [np.zeros(e.shape, bool)]    # undecidable at step k (doji / missing candle)
            for step in range(1, MAXK + 1):
                x, xv = fd[idx + step], fv[idx + step]
                win_lv.append(alive & xv & (x == s))
                skip_at.append(alive & ~(xv & (x != 0)))
                alive = alive & xv & (x == -s)
                alive_after.append(alive)
            modes = {}
            for k in range(MAXK + 1):
                wins = np.zeros(e.shape, bool)
                for lv in range(k + 1):
                    wins |= win_lv[lv]
                skipped = np.zeros(e.shape, bool)
                for st_ in range(1, k + 1):
                    skipped |= skip_at[st_]
                out = np.zeros(e.shape, np.int8)
                out[wins] = 1
                out[alive_after[k]] = -1
                modes[k] = _pack(out, [int(win_lv[lv].sum()) for lv in range(k + 1)], int(skipped.sum()), days, d0, HOURS)
            res["sides"][name] = dict(m=modes)
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
        hours = hours_for(pair)
        days = pick_days(dirm, val, tf, months, hours)
        if len(days) == 0:
            return None
        mins = sorted(minutes)
        return dict(pair=pair, tf=tf, months=months, d0=d0, dirm=dirm, val=val, days=days, minutes=mins, hours=hours,
                    stats=[minute_stats(dirm, val, tf, m, days, d0, hours) for m in mins])


    def day_date(d0, i):
        return dt.date(1970, 1, 1) + dt.timedelta(days=int(d0 + i))


    # ------------------------------------------------------------------ CSV
    def _letter(dv, vv):
        return ("G" if dv == 1 else "R" if dv == -1 else "D") if vv else ""


    def make_csv(res, kmax=MAXK):
        """-> (grid_csv bytes, summary_csv bytes). Grid: one row per day+hour, one column per minute slot."""
        tf, per = res["tf"], 60 // res["tf"]
        g = io.StringIO()
        w = csv.writer(g)
        w.writerow(["date", "weekday", "hour"] + ["%02d" % (k * tf) for k in range(per)])
        for i in res["days"]:
            dd = day_date(res["d0"], i)
            for h in res.get("hours", HOURS):
                w.writerow([dd.isoformat(), WD[dd.weekday()], "%02d" % h] +
                           [_letter(res["dirm"][i, h * per + k], res["val"][i, h * per + k]) for k in range(per)])
        s = io.StringIO()
        w = csv.writer(s)
        w.writerow(["pair", "tf_min", "minute", "days", "green", "red", "doji", "side", "mode", "win", "loss", "winrate_pct",
                    "win_direct", "win_mtg1", "win_mtg2", "win_mtg3", "step_skipped", "b2b_loss_day", "b2b_loss_day_when",
                    "b2b_loss_chain", "b2b_loss_chain_when"])
        for st in res["stats"]:
            for side in ("G", "R"):
                for k in range(kmax + 1):
                    x = st["sides"][side]["m"][k]
                    lv = x["levels"] + [""] * (MAXK + 1 - len(x["levels"]))
                    w.writerow([res["pair"], tf, "%02d" % st["minute"], st["days"], st["g"], st["r"], st["doji"],
                                "GREEN" if side == "G" else "RED", MODES[k], x["win"], x["loss"], "%.1f" % (100 * x["wr"]),
                                lv[0], lv[1], lv[2], lv[3], x["skipped"], x["b2b_day"], x["b2b_day_at"], x["b2b_chain"], x["b2b_chain_at"]])
        return g.getvalue().encode("utf-8-sig"), s.getvalue().encode("utf-8-sig")


    # ------------------------------------------------------------------ PDF
    def make_pdf(res, kmax=MAXK):
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import Flowable, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

        tf, per = res["tf"], 60 // res["tf"]
        sel = set(m // tf for m in res["minutes"])
        pair = res["pair"].replace("_", "/")
        GREEN, RED, GREY = colors.Color(0.0, 0.5, 0.25), colors.Color(0.8, 0.15, 0.2), colors.Color(0.45, 0.45, 0.5)
        NAVY, LIGHT, ZEBRA = colors.Color(0.09, 0.22, 0.4), colors.Color(0.86, 0.91, 0.96), colors.Color(0.94, 0.96, 0.98)
        LINE = colors.Color(0.8, 0.84, 0.9)
        page = landscape(A4)
        M = 20
        width = page[0] - 2 * M
        HRS = res.get("hours", HOURS)

        class DayGrid(Flowable):
            RH, HH, TH, LW = 9.6, 10.5, 12.0, 26.0

            def __init__(self, i):
                super().__init__()
                self.i = i
                self.width = width
                self.height = self.TH + self.HH + len(HRS) * self.RH + 8

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
                for r, h in enumerate(HRS):
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
        h2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=10.5, textColor=NAVY, spaceBefore=6, spaceAfter=3)
        p9 = ParagraphStyle("p9", fontName="Helvetica", fontSize=8.5, leading=11)
        pc = ParagraphStyle("pc", fontName="Helvetica", fontSize=6.6, leading=7.8, alignment=0)

        def when(x, key):
            return Paragraph(x[key] or "-", pc)

        story = [Paragraph("%s  -  Engine 3 Minute Grid  (M%d)" % (pair, tf), h1)]
        dts = [day_date(res["d0"], i) for i in res["days"]]
        story.append(Paragraph(
            "Period: %s to %s  (%d trading days, last %s)   |   Entry hours %s BDT (UTC+6)   |   Entry minutes: %s" % (
                dts[0], dts[-1], len(dts), period_label_en(res["months"]), hours_label(HRS),
                ", ".join(":%02d" % m for m in res["minutes"])), p9))
        story.append(Paragraph(
            "<b>How to read:</b> GREEN block = you trade GREEN at that minute, RED block = you trade RED. The light-green block is the colour that appears more often at that minute. "
            "<b>No-MTG</b> win = the entry candle goes your way. <b>MTG1</b> = the 2nd candle may win it, <b>MTG2</b> = the 3rd candle, <b>MTG3</b> = the 4th candle "
            "(every earlier candle went the wrong way). Loss = all candles of that mode went the wrong way. "
            "Win breakdown = how many wins came directly, then from MTG step 1, 2, 3. "
            "Doji (open = close) is shown as D and is never a loss: a doji entry, or an MTG step landing on a doji, is skipped (not counted). "
            "B2B = longest run of back-to-back losses in hour order: day (restarts every day) / chain (continues across days); the two right columns say WHEN it first happened (BDT date and hour). "
            "Daily tables at the end: orange header = your selected minute(s).", p9))
        story.append(Spacer(1, 4))
        head = ["Trade", "Mode", "Win", "Win breakdown (direct + MTG steps)", "Loss", "WR %", "B2B\nday / chain",
                "Longest loss run - day (when)", "Longest loss run - chain (when)"]
        cw_ = [40, 46, 36, 150, 36, 40, 50, 204, 200]
        for st in res["stats"]:
            tot = st["g"] + st["r"]
            gp = 100.0 * st["g"] / tot if tot else 0.0
            rows = [head]
            style = [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                     ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, 0), 7),
                     ("FONTSIZE", (0, 1), (-1, -1), 7.6), ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                     ("ALIGN", (0, 0), (6, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                     ("GRID", (0, 0), (-1, -1), 0.4, LINE), ("TOPPADDING", (0, 1), (-1, -1), 1.5),
                     ("BOTTOMPADDING", (0, 1), (-1, -1), 1.5)]
            for side in ("G", "R"):
                r0 = len(rows)
                for k in range(kmax + 1):
                    x = st["sides"][side]["m"][k]
                    bd = " + ".join(str(v) for v in x["levels"][:kmax + 1])
                    rows.append(["GREEN" if side == "G" else "RED", MODES[k], x["win"], bd, x["loss"], "%.1f" % (100 * x["wr"]),
                                 "%d / %d" % (x["b2b_day"], x["b2b_chain"]), when(x, "b2b_day_at"), when(x, "b2b_chain_at")])
                    r = len(rows) - 1
                    style.append(("FONTNAME", (5, r), (5, r), "Helvetica-Bold"))
                    if side == st["dominant"]:
                        style.append(("BACKGROUND", (1, r), (-1, r), colors.Color(0.93, 0.97, 0.9)))
                    elif k % 2 == 1:
                        style.append(("BACKGROUND", (1, r), (-1, r), ZEBRA))
                style.append(("SPAN", (0, r0), (0, r0 + kmax)))
                style.append(("TEXTCOLOR", (0, r0), (0, r0), GREEN if side == "G" else RED))
                style.append(("FONTNAME", (0, r0), (0, r0), "Helvetica-Bold"))
                style.append(("LINEABOVE", (0, r0), (-1, r0), 0.9, NAVY))
            t = Table(rows, repeatRows=1, colWidths=cw_)
            t.setStyle(TableStyle(style))
            title = Paragraph("Minute :%02d   -   %d days   |   Green %d (%.0f%%)   Red %d (%.0f%%)   Doji %d" % (
                st["minute"], st["days"], st["g"], gp, st["r"], 100 - gp if tot else 0.0, st["doji"]), h2)
            story.append(KeepTogether([title, t]))
        story.append(PageBreak())
        story.append(Paragraph("Daily candle tables (G = green, R = red, D = doji)", h2))
        for i in res["days"]:
            story.append(KeepTogether([DayGrid(i)]))
        bio = io.BytesIO()
        doc = SimpleDocTemplate(bio, pagesize=page, leftMargin=M, rightMargin=M, topMargin=M, bottomMargin=M,
                                title="%s M%d minute grid" % (pair, tf))
        doc.build(story)
        bio.seek(0)
        return bio


    # ------------------------------------------------------------------ Telegram text (Bengali)
    def bn_text(res, limit=6):
        out = []
        for st in res["stats"][:limit]:
            tot = st["g"] + st["r"]
            gp = round(100 * st["g"] / tot) if tot else 0
            out.append(":%02d  (%d দিন • গ্রিন %d%% • রেড %d%% • ডোজি %d)" % (st["minute"], st["days"], gp, 100 - gp if tot else 0, st["doji"]))
            for side, icon, nm in (("G", "🟢", "গ্রিন"), ("R", "🔴", "রেড")):
                ms = st["sides"][side]["m"]
                star = " ⭐" if side == st["dominant"] else ""
                out.append("%s %s%s: " % (icon, nm, star) + " • ".join(
                    "%s %.0f%%" % (["নন-MTG", "MTG1", "MTG2", "MTG3"][k], 100 * ms[k]["wr"]) for k in range(MAXK + 1)))
                out.append("    B2B লস (দিনে/চেইন): " + " • ".join(
                    "%s %d/%d" % (["নন", "M1", "M2", "M3"][k], ms[k]["b2b_day"], ms[k]["b2b_chain"]) for k in range(MAXK + 1)))
                out.append("    📅 চেইন কবে: " + " • ".join(
                    "%s %s" % (["নন", "M1", "M2", "M3"][k], ms[k]["b2b_chain_at"] or "-") for k in range(MAXK + 1)))
            out.append("")
        if len(res["stats"]) > limit:
            out.append("… বাকি মিনিটগুলো PDF/CSV-তে।")
        return "\n".join(out)


    _ns = dict(locals())
    return types.SimpleNamespace(**{k: v for k, v in _ns.items() if not k.startswith("__")})


engine3 = _build_engine3()


# ==========================================================================
# EMBEDDED COPY of engine4.py (identical code, private to this file)
# ==========================================================================
def _build_engine4():
    def _g():
        return _ns

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

    e3 = engine3

    HOURS = e3.HOURS
    MAXK = e3.MAXK
    MODES = e3.MODES
    SIDE_NAME = {"G": "GREEN", "R": "RED"}


    def outcomes(dirm, val, tf, minute, days, side, k, hours=None):
        """-> (out (days,hours) int8: 1 win, -1 loss, 0 no data ; lvl int8: win level 0..k, -1 otherwise ; dj bool: the win came from a doji)."""
        per = 60 // tf
        S = 1440 // tf
        j = minute // tf
        s = 1 if side == "G" else -1
        HOURS = hours or _g()["HOURS"]
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
        hours = e3.hours_for(pair)
        days = e3.pick_days(dirm, val, tf, months, hours)
        if len(days) == 0:
            return None
        rows = []
        for m in sorted(minutes):
            for side in sides:
                out, lvl, dj = outcomes(dirm, val, tf, m, days, side, k, hours)
                bw, bl = int((out == 1).sum()), int((out == -1).sum())
                sigs, miss = find_signals(out, lvl, dj, n_loss, scope, atleast)
                sw = sum(1 for x in sigs if x[2] == 1)
                sl = len(sigs) - sw
                levels = [sum(1 for x in sigs if x[2] == 1 and x[3] == L) for L in range(k + 1)]
                rows.append(dict(minute=m, side=side, base_win=bw, base_loss=bl, base_dj=int((dj & (out == 1)).sum()),
                                 base_wr=(bw / (bw + bl) if bw + bl else 0.0),
                                 sig=len(sigs), win=sw, loss=sl, wr=(sw / len(sigs) if sigs else 0.0),
                                 doji_win=sum(1 for x in sigs if x[2] == 1 and x[4]), missing=miss, levels=levels,
                                 signals=[(int(days[i]), hours[j], v, L, d_) for i, j, v, L, d_ in sigs]))
        return dict(pair=pair, tf=tf, months=months, d0=d0, days=days, hours=hours, k=k, n_loss=n_loss, scope=scope, atleast=atleast, rows=rows)


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
        story.append(flow("Period: %s to %s (%d trading days, last %s)  |  Entry hours %s BDT (UTC+6)  |  %s" % (dts[0], dts[-1], len(dts), per_en, e3.hours_label(res["hours"]), rule_en),
                          "সময়কাল: %s থেকে %s (%d ট্রেডিং-দিন, শেষ %s)  |  ঘণ্টা %s (UTC+6)  |  %s" % (dts[0], dts[-1], len(dts), per_bn, e3.hours_label(res["hours"]), rule_bn), 8.5))
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


    _ns = dict(locals())
    return types.SimpleNamespace(**{k: v for k, v in _ns.items() if not k.startswith("__")})


engine4 = _build_engine4()


# ==========================================================================
# EMBEDDED COPY of engine6.py (identical code, private to this file)
# ==========================================================================
def _build_engine6():
    def _g():
        return _ns

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

    e3 = engine3

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
        HRS = res.get("hours", HOURS)
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
                self.height = self.TH + self.HH + len(HRS) * self.RH + 8

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
                for r, h in enumerate(HRS):
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
            "Period: %s to %s  (%d trading days, last %s)   |   Entry hours %s BDT (UTC+6)   |   Entry minutes: %s" % (
                dts[0], dts[-1], len(dts), e3.period_label_en(res["months"]), e3.hours_label(HRS),
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


    _ns = dict(locals())
    return types.SimpleNamespace(**{k: v for k, v in _ns.items() if not k.startswith("__")})


engine6 = _build_engine6()


# ==========================================================================
# EMBEDDED COPY of engine8.py (identical code, private to this file)
# ==========================================================================
def _build_engine8():
    def _g():
        return _ns

    """Engine 8 — Engine 6's numbers, but sent as plain Telegram TEXT (no PDF, no buttons, no database, no Telegram code here).

    Engine 8 reuses engine6 / engine3 1:1 (same data, same candle / MTG / doji rules, same B2B day / chain numbers, same ranking).
    Only the output differs: for every pair / MTG step we return one short text with two headings, GREEN and RED, and under each
    heading the minutes that share the LOWEST one-day back-to-back loss.
    """
    pass  # engine6 is embedded below

    MAXK = engine6.MAXK
    MODES = engine6.MODES
    LIMIT = 4000        # Telegram's hard limit is 4096 characters per message


    def analyze(pair, ts, o, c, tf, months, minutes):
        """Same as Engine 6 / Engine 3."""
        return engine6.analyze(pair, ts, o, c, tf, months, minutes)


    def collect(res, levels):
        """-> {k: {'G': ranked rows, 'R': ranked rows}} for every chosen MTG level (rows already sorted, rank 1 first)."""
        return {k: {s: engine6.ranked(res, k, s) for s in ("G", "R")} for k in sorted(set(levels))}


    def lowest(rows):
        """-> (lowest one-day B2B loss, [minutes that have it, in rank order]); (None, []) when there are no rows."""
        if not rows:
            return None, []
        lo = rows[0]["b2b_day"]          # rows are sorted by b2b_day first
        return lo, [r["minute"] for r in rows if r["b2b_day"] == lo]


    def minutes_line(rows):
        """-> (lowest one-day B2B loss, ':05 :12 :30') for the minutes that share it, sorted by time; (None, '—') when no data."""
        lo, mins = lowest(rows)
        if lo is None:
            return None, "—"
        return lo, " ".join(":%02d" % m for m in sorted(mins))


    def side_block(icon, name, rows):
        lo, line = minutes_line(rows)
        title = "%s %s" % (icon, name) if lo is None else "%s %s • B2B ডে লস: %d" % (icon, name, lo)
        return "%s\n%s" % (title, line)


    def messages(head, rows_by_k, labels):
        """-> list of message texts (each <= LIMIT chars) for one pair.

        head: first line (pair / time frame / period); rows_by_k: collect() output; labels: {k: 'MTG1', ...}.
        """
        blocks = []
        for k in sorted(rows_by_k):
            d = rows_by_k[k]
            blocks.append("%s\n\n%s\n\n%s" % (labels.get(k, MODES[k]), side_block("🟢", "গ্রিন", d["G"]), side_block("🔴", "রেড", d["R"])))
        out, cur = [], head
        for b in blocks:
            nxt = cur + "\n\n━━━━━━━━━━\n" + b
            if len(nxt) > LIMIT and cur != head:
                out.append(cur)
                cur = head + "\n\n━━━━━━━━━━\n" + b
            else:
                cur = nxt
        out.append(cur)
        return [t[:LIMIT] for t in out]


    _ns = dict(locals())
    return types.SimpleNamespace(**{k: v for k, v in _ns.items() if not k.startswith("__")})


engine8 = _build_engine8()


# ==========================================================================
# EMBEDDED COPY of engine9.py (identical code, private to this file)
# ==========================================================================
def _build_engine9():
    def _g():
        return _ns

    """Engine 9 — Engine 8 + Engine 4 in one step (no database, no Telegram code here).

    1) Engine 8 part: for every pair, find the minutes that share the LOWEST one-day back-to-back loss at the chosen MTG,
       separately for GREEN and for RED (same data / candle / MTG / doji rules as Engine 3 / 6 / 8).
    2) Engine 4 part: take exactly those minutes (GREEN minutes -> GREEN trades, RED minutes -> RED trades) and run the
       "signal after N back-to-back losses" counting of Engine 4 on them (doji = win, same as Engine 4).
    Output is a short Telegram text: per side and per minute -> signals, win, loss, win rate, doji wins, plus a total line.
    """
    e3 = engine3
    pass  # engine4 is embedded below
    pass  # engine6 is embedded below
    pass  # engine8 is embedded below

    MODES = engine6.MODES
    HOURS = e3.HOURS
    LIMIT = 4000        # Telegram's hard limit is 4096 characters per message


    def analyze(pair, ts, o, c, tf, months, ks, n_loss, scope, atleast):
        """ks = list of MTG levels (0..3). -> dict or None when there is not enough data."""
        ks = sorted(set(ks))
        res = engine6.analyze(pair, ts, o, c, tf, months, list(range(0, 60, tf)))
        if res is None:
            return None
        coll = engine8.collect(res, ks)
        per = {}
        for k in ks:
            sides = {}
            for side in ("G", "R"):
                lo, mins = engine8.lowest(coll[k][side])
                rows = []
                for m in sorted(mins):
                    out, lvl, dj = engine4.outcomes(res["dirm"], res["val"], tf, m, res["days"], side, k, res["hours"])
                    sigs, miss = engine4.find_signals(out, lvl, dj, n_loss, scope, atleast)
                    sw = sum(1 for x in sigs if x[2] == 1)
                    rows.append(dict(minute=m, sig=len(sigs), win=sw, loss=len(sigs) - sw,
                                     doji=sum(1 for x in sigs if x[2] == 1 and x[4]), missing=miss, signals=sigs))
                sides[side] = dict(lo=lo, rows=rows)
            per[k] = sides
        return dict(pair=pair, tf=tf, months=months, d0=res["d0"], days=res["days"], hours=res["hours"], ks=ks, n_loss=n_loss,
                    scope=scope, atleast=atleast, per=per)


    def _wr(w, n):
        return "%.0f%%" % (100.0 * w / n) if n else "-"


    def side_block(icon, name, d):
        if d["lo"] is None or not d["rows"]:
            return "%s %s\nডেটা নেই" % (icon, name)
        lines = ["%s %s • B2B ডে লস: %d" % (icon, name, d["lo"])]
        for r in d["rows"]:
            if r["sig"] == 0:
                lines.append(":%02d → 📡0" % r["minute"])
            else:
                lines.append(":%02d → 📡%d ✅%d ❌%d • %s • ডোজি %d" % (
                    r["minute"], r["sig"], r["win"], r["loss"], _wr(r["win"], r["sig"]), r["doji"]))
        sig = sum(r["sig"] for r in d["rows"])
        win = sum(r["win"] for r in d["rows"])
        loss = sum(r["loss"] for r in d["rows"])
        dj = sum(r["doji"] for r in d["rows"])
        lines.append("মোট → 📡%d ✅%d ❌%d • %s • ডোজি %d" % (sig, win, loss, _wr(win, sig), dj))
        return "\n".join(lines)


    def messages(head, res, labels, scope_txt):
        """-> list of message texts (each <= LIMIT chars) for one pair (one block per chosen MTG level)."""
        top = "%s\nপরপর %d লস (%s, %s) • ডোজি = উইন" % (
            head, res["n_loss"], "N বা বেশি" if res["atleast"] else "ঠিক N", scope_txt)
        blocks = []
        for k in res["ks"]:
            d = res["per"][k]
            blocks.append("%s\n\n%s\n\n%s" % (labels.get(k, MODES[k]), side_block("🟢", "গ্রিন", d["G"]), side_block("🔴", "রেড", d["R"])))
        out, cur = [], top
        for b in blocks:
            nxt = cur + "\n\n━━━━━━━━━━\n" + b
            if len(nxt) > LIMIT and cur != top:
                out.append(cur)
                cur = top + "\n\n━━━━━━━━━━\n" + b
            else:
                cur = nxt
        out.append(cur)
        return [t[:LIMIT] for t in out]


    _ns = dict(locals())
    return types.SimpleNamespace(**{k: v for k, v in _ns.items() if not k.startswith("__")})


engine9 = _build_engine9()





# ==========================================================================
# ENGINE 13 — own code starts here (everything above is the embedded engine3/4/6/8/9, identical to engine12's copy)
# ==========================================================================
import asyncio
import csv
import datetime as dt
import html
import io
import logging
import math
import os
import re
import sys
import time

import numpy as np
from telegram import BotCommand, InlineKeyboardButton as Btn, InlineKeyboardMarkup as Markup
from telegram.error import BadRequest
from telegram.ext import ApplicationHandlerStop, CallbackQueryHandler, CommandHandler, MessageHandler, filters

import pairs
import style
import source

log = logging.getLogger("engine13")
BDT = pairs.BDT
OFF = engine3.OFF
MAX_DAYS = 14                      # most test days in one run (the walk-forward test gets better with more days)
E3_MAX_DAYS = 365
WAIT_SECS = 900                    # a ✏️ prompt that nobody answers expires after 15 minutes
SUP = {1: "¹", 2: "²", 3: "³"}
MTG_LB = {0: "নন-MTG", 1: "MTG1", 2: "MTG2", 3: "MTG3"}
MODE_LB = {"exact": "ঠিক N", "atleast": "N বা বেশি"}
FMODE_LB = {"with": "ট্রেন্ডের সাথে", "counter": "ট্রেন্ডের উল্টো"}
TF_PRESETS = (1, 3, 5)
MONTH_PRESETS = (3, 5, 6)
MINSIG_PRESETS = (1, 3, 5)
FAST_GRID = (3, 5, 8, 9, 13, 21)   # the same EMA grid as /filter of the OTC bot
SLOW_GRID = (21, 34, 50, 89, 100, 200)
MAX_PERIOD = 300
PAYOUT = float(os.environ.get("PAYOUT", "80")) / 100.0
D = {}                             # injected by register(): send, allowed, deny, lock, bot

# wait: False | "tf" | "months" | "minsig" | "nloss" | "dates" | "grid"
# days: "y" (yesterday) | "t" (today) | "custom" (E["dates"] = list of ISO dates)
E = dict(mkt="real", tf=1, months=4, mtg=[1], n=3, mode="exact", minsig=3, pairs=[], page="main", wait=False, wait_at=0.0,
         days="y", dates=[], show_list=False, show_kept=True, csv=True, fast=list(FAST_GRID), slow=list(SLOW_GRID),
         fmodes=["with", "counter"], running=False, stop=False)


# ---------------------------------------------------------------- small helpers (copied from engine12, so nothing is imported from it)
def _digits(s):
    return (s or "").translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789"))
def _tick(on, label):
    return ("✅ " if on else "") + label
def months_label(months):
    d = engine3.period_days(months)
    if d == 365:
        return "1Y"
    if d % 30 == 0 and d != 360:
        return "%dM" % (d // 30)
    return "%d দিন" % d
def parse_period(txt):
    """'4' / '4M' = months, '45D' / '45 দিন' = days, '1Y' = year  ->  months (number) or None."""
    t = txt.strip().lower().replace(" ", "")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(d|day|days|দিন|m|mo|month|months|মাস|y|yr|year|years|বছর)?", t)
    if not m:
        return None
    n, unit = float(m.group(1)), (m.group(2) or "m")
    if unit.startswith(("d", "দ")):
        days = n
    elif unit.startswith(("y", "ব")):
        days = n * 365
    else:
        days = n * 30
    days = int(round(days))
    if days < 1 or days > E3_MAX_DAYS:
        return None
    if days == 365:
        return 12
    return days // 30 if days % 30 == 0 else days / 30.0
def data_missing(plist):
    if any(not pairs.is_otc(p) for p in plist) and not source.enabled():
        return "DATABASE_URL_1/2/3 সেট করা নেই (রিয়েল মার্কেটের ক্যান্ডেল ডেটাবেজ)।"
    if any(pairs.is_otc(p) for p in plist) and not source.otc_enabled():
        return "OTC_DATABASE_URL_1/2/3 সেট করা নেই (OTC মার্কেটের ক্যান্ডেল ডেটাবেজ)।"
    return None
def day_ts(d):
    """BDT calendar date -> unix seconds of its 00:00 (UTC+6)."""
    return int(dt.datetime(d.year, d.month, d.day, tzinfo=BDT).timestamp())
def _day_of_month(d, today):
    try:
        c = dt.date(today.year, today.month, d)
    except ValueError:
        c = None
    if c is None or c > today:
        y, m = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
        c = dt.date(y, m, d)
    return c
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
def test_dates():
    today = dt.datetime.now(BDT).date()
    if E["days"] == "y":
        return [today - dt.timedelta(days=1)]
    if E["days"] == "t":
        return [today]
    return sorted(dt.date.fromisoformat(x) for x in E["dates"])
def dates_label(ds=None):
    ds = ds if ds is not None else test_dates()
    if not ds:
        return "— (তারিখ এখনও দাওনি)"
    if len(ds) == 1:
        return ds[0].strftime("%d %b")
    if all((b - a).days == 1 for a, b in zip(ds, ds[1:])):          # 14,15,16,17,18 -> "14–18 Nov"
        a, b = ds[0], ds[-1]
        if (a.year, a.month) == (b.year, b.month):
            return "%d–%d %s" % (a.day, b.day, b.strftime("%b"))
        return "%s – %s" % (a.strftime("%d %b"), b.strftime("%d %b"))
    if len({(d.year, d.month) for d in ds}) == 1:                    # 14,16,18 -> "14,16,18 Nov"
        return ",".join(str(d.day) for d in ds) + " " + ds[0].strftime("%b")
    return ", ".join(d.strftime("%d %b") for d in ds)


# ---------------------------------------------------------------- 1) list rules (same as Engine 12 / Engine 11)
def rules_from(res, minsig):
    """Engine 9 result of one pair -> radar rules (100% win ratio and >= minsig signals)."""
    best = {}
    for k in res["ks"]:                                   # lowest MTG first -> it wins when several qualify
        for side in ("G", "R"):
            for r in res["per"][k][side]["rows"]:
                if r["sig"] >= minsig and r["win"] == r["sig"]:
                    best.setdefault((r["minute"], side), dict(
                        pair=res["pair"], minute=r["minute"], side=side, mtg=k, tf=res["tf"], n=res["n_loss"],
                        atleast=bool(res["atleast"]), sig=r["sig"], win=r["win"], doji=r["doji"]))
    return sorted(best.values(), key=lambda x: (x["side"] != "G", x["minute"]))
def _mins(rs, multi):
    return " ".join(":%02d%s" % (x["minute"], ("·M%d" % x["mtg"]) if multi else "") for x in rs) or "—"
def _compose(head, entries):
    """entries: [(pair name, rules or None)] -> list of messages (<= 3900 chars). Per pair: name + green minutes + red minutes;
    the very last block merges the minutes of all pairs, in serial order, each minute once."""
    multi = len({x["mtg"] for _, rs in entries if rs for x in rs}) > 1
    blocks, allg, allr = [], set(), set()
    for name, rs in entries:
        if rs is None:
            blocks.append("💱 %s — ডেটা নেই/কম" % name)
            continue
        g = [x for x in rs if x["side"] == "G"]
        d = [x for x in rs if x["side"] == "R"]
        allg |= {x["minute"] for x in g}
        allr |= {x["minute"] for x in d}
        blocks.append("💱 %s\n🟢 %s\n🔴 %s" % (name, _mins(g, multi), _mins(d, multi)))
    fmt = lambda s_: " ".join(":%02d" % m for m in sorted(s_)) or "—"  # noqa: E731
    tail = ("━━━━━━━━━━━━\nসব পেয়ার মিলিয়ে (সিরিয়াল অনুযায়ী, প্রতিটা মিনিট একবার)\n\n"
            "🟢 UP (%d টা মিনিট)\n%s\n\n🔴 DOWN (%d টা মিনিট)\n%s" % (len(allg), fmt(allg), len(allr), fmt(allr)))
    out, cur = [], head
    for b in blocks:
        if len(cur) + len(b) + 2 > 3900:
            out.append(cur)
            cur = b
        else:
            cur += "\n\n" + b
    if len(cur) + len(tail) + 2 > 3900:
        out.append(cur)
        cur = tail
    else:
        cur += "\n\n" + tail
    out.append(cur)
    return out


# ---------------------------------------------------------------- 2) EMA filter (same rules as /filter of the OTC bot, filt.py)
def ema_list(arr, n):
    """Same as filt.ema_series: first value seeds the EMA, empty candles are skipped."""
    k = 2.0 / (n + 1)
    out, prev = [], None
    for v in arr:
        if v is not None:
            prev = v if prev is None else prev + k * (v - prev)
        out.append(prev)
    return out


def _day_ctx(ts, c, d, tf, periods):
    """M1 closes of day d-1 + d (the earlier day only warms the EMAs up) -> TF closes, count of candles so far, EMA series per period."""
    lo = day_ts(d) - 86400
    m = (ts >= lo) & (ts < lo + 2 * 86400)
    arr = [None] * 2880
    for t_, v in zip(ts[m].tolist(), c[m].tolist()):
        arr[int((t_ - lo) // 60)] = float(v)
    ser = [arr[k * tf + tf - 1] for k in range(2880 // tf)]
    cnt, n = [], 0
    for v in ser:
        n += v is not None
        cnt.append(n)
    return ser, cnt, {p: ema_list(ser, p) for p in periods}


def usable(s, ns):
    return s.get("px") is not None and s["cnt"] >= ns


def trend(s, nf, ns):
    """1 = price above both EMAs, -1 = below both, 0 = in between (price = close of the last closed candle before the entry)."""
    ef, es, p = s["ema"].get(nf), s["ema"].get(ns), s["px"]
    if ef is None or es is None:
        return 0
    if p > ef and p > es:
        return 1
    if p < ef and p < es:
        return -1
    return 0


def keeps(s, tr, fm):
    want = 1 if s["side"] == "G" else -1
    return tr == (want if fm == "with" else -want)


def tally(items):
    """-> n, wins, losses, {win level: count}, P/L in stake units (doubling MTG, payout from env PAYOUT, default 80%)."""
    w = l = 0
    lv, pl = {}, 0.0
    for s in items:
        if s["res"] == 1:
            w += 1
            e = s["lvl"]
            lv[e] = lv.get(e, 0) + 1
            pl += PAYOUT * 2 ** e - (2 ** e - 1)
        else:
            l += 1
            pl -= 2 ** (s["mtg"] + 1) - 1
    return dict(n=w + l, w=w, l=l, lv=lv, pl=pl)


def wilson(w, n, z=1.96):
    if n == 0:
        return 0.0
    p = w / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def research(items, fast, slow, fmodes):
    """Every Fast/Slow pair of the grid, with / counter trend -> rows ranked by the Wilson lower bound (same as /filter 'best combo')."""
    rows = []
    for nf in fast:
        for ns in slow:
            if nf >= ns:
                continue
            ev = [s for s in items if usable(s, ns)]
            b = tally(ev)
            if b["n"] < 10:
                continue
            tr = {id(s): trend(s, nf, ns) for s in ev}
            for fm in fmodes:
                kept = [s for s in ev if keeps(s, tr[id(s)], fm)]
                k = tally(kept)
                if k["n"] < max(8, math.ceil(0.25 * b["n"])):
                    continue
                rows.append(dict(nf=nf, ns=ns, fm=fm, k=k, b=b, lb=wilson(k["w"], k["n"]),
                                 wr=k["w"] * 100.0 / k["n"], base=b["w"] * 100.0 / b["n"]))
    rows.sort(key=lambda r: (-r["lb"], -r["wr"]))
    return rows


def walk_forward(sigs, dates, st):
    """For every test day: pick the best EMA combo using ONLY the signals of the earlier test days, then apply it to that day.
    The first day has nothing to learn from, so it gets no filter. -> [dict(date, combo|None, all, kept, removed)]"""
    by = {d: [s for s in sigs if s["date"] == d] for d in dates}
    out = []
    for i, d in enumerate(dates):
        row = dict(date=d, combo=None, all=by[d], kept=[], removed=[])
        train = [s for dd in dates[:i] for s in by[dd]]
        rows = research(train, st["fast"], st["slow"], st["fmodes"]) if train else []
        if rows:
            r = rows[0]
            row["combo"] = r
            for s in by[d]:
                ok = usable(s, r["ns"]) and keeps(s, trend(s, r["nf"], r["ns"]), r["fm"])
                (row["kept"] if ok else row["removed"]).append(s)
        out.append(row)
    return out


def parse_grid(txt):
    """'5,13,21 / 34,50,100' (Fast list / Slow list) or '13/50' -> (fast, slow) or None."""
    parts = [p for p in re.split(r"[/|;]", txt) if p.strip()]
    if len(parts) != 2:
        return None
    lists = []
    for p in parts:
        nums = sorted({int(x) for x in re.findall(r"\d+", p)})
        if not nums or nums[0] < 2 or nums[-1] > MAX_PERIOD or len(nums) > 8:
            return None
        lists.append(nums)
    fast, slow = lists
    if not any(a < b for a in fast for b in slow):
        return None
    return fast, slow


# ---------------------------------------------------------------- 3) the work for ONE pair: a list for every test day (data up to the day before) + replay of that day
def compute_pair(pair, dates, st):
    """Reads the pair's M1 candles once. For every test day D: (a) the back-test list is built ONLY from candles before 00:00 of D
    (so the test day can never leak into its own list), exactly like Engine 11 with 'custom date = D-1'; (b) that list is replayed on
    day D with the same code as /otctest. -> dict(pair, lists{date: dict|None}, sigs, no_data[date], partial[(date, hh:mm)], missing)."""
    tf, n, atleast = st["tf"], st["n"], st["mode"] == "atleast"
    days_n = engine3.period_days(st["months"])
    out = dict(pair=pair, lists={}, sigs=[], no_data=[], partial=[], missing=0)
    lo_all = day_ts(dates[0]) - (days_n + 10) * 86400
    hi_all = day_ts(dates[-1]) + 86400 + 2 * 3600                  # 2 more hours: MTG steps of the last candles
    ts, o, c = source.load_oc(pair, lo_all, hi_all)
    if len(ts) == 0:
        out["no_data"] = list(dates)
        out["lists"] = {d: None for d in dates}
        return out
    ts, o, c = np.asarray(ts), np.asarray(o), np.asarray(c)
    day_rules = {}
    for d in dates:
        hi = day_ts(d)                                              # candles strictly BEFORE the test day
        lo = hi - (days_n + 10) * 86400
        msk = (ts >= lo) & (ts < hi)
        res = None
        if msk.any():
            res = engine9.analyze(pair, ts[msk], o[msk], c[msk], tf, st["months"], sorted(st["mtg"]), n, "day", atleast)
        if res is None:
            out["lists"][d] = None
            day_rules[d] = []
            continue
        first_ts = int(ts[msk][0])
        first_have = int((first_ts + OFF) // 86400) - res["d0"]
        first_want = int(res["days"][-1]) - days_n + 1
        short = None
        if first_have - first_want > 3:                             # > 3 days missing (weekend gaps are not counted)
            short = dict(want=days_n, got=int(res["days"][-1]) - int(res["days"][0]) + 1)
        rules = rules_from(res, st["minsig"])
        out["lists"][d] = dict(rules=rules, days=len(res["days"]), short=short,
                               first=engine3.day_date(res["d0"], res["days"][0]), last=engine3.day_date(res["d0"], res["days"][-1]))
        day_rules[d] = rules
    # ---- replay (copy of otctest.replay, one pair)
    lt, dd = engine3.to_tf(ts, o, c, tf)
    if len(lt) == 0:
        out["no_data"] = [d for d in dates if day_rules.get(d)]
        return out
    d0, dirm, val = engine3.build_grid(lt, dd, tf)
    nd = dirm.shape[0]
    hours = engine3.hours_for(pair)
    for d in dates:
        rs = day_rules.get(d) or []
        if not rs:
            continue
        row = (day_ts(d) + OFF) // 86400 - d0
        if row < 0 or row >= nd or not val[row].any():
            out["no_data"].append(d)
            continue
        last_loc = int(lt[(lt // 86400) == (day_ts(d) + OFF) // 86400].max())
        if last_loc < day_ts(d) + OFF + 23 * 3600:                  # candles end before the 23:00 hour -> day not complete
            out["partial"].append((d, dt.datetime.fromtimestamp(last_loc - OFF, BDT).strftime("%H:%M")))
        for r in rs:
            res_, lvl, dj = engine4.outcomes(dirm, val, tf, r["minute"], np.array([row]), r["side"], r["mtg"], hours)
            streak, loss_h = 0, []
            for j, h in enumerate(hours):
                v = int(res_[0, j])
                due = streak >= n if atleast else streak == n
                if v == 0:
                    if due:
                        out["missing"] += 1
                    continue
                if due:
                    out["sigs"].append(dict(ts=day_ts(d) + h * 3600 + r["minute"] * 60, date=d, pair=pair, side=r["side"], minute=r["minute"],
                                            hour=h, mtg=r["mtg"], res=v, lvl=int(lvl[0, j]), doji=bool(dj[0, j]), streak=streak,
                                            loss_hours=list(loss_h)))
                if v == -1:
                    streak += 1
                    loss_h.append(h)
                else:
                    streak, loss_h = 0, []
    periods = sorted(set(st["fast"]) | set(st["slow"]))
    cache = {}
    for s in out["sigs"]:                                           # EMA context of every signal (same rules as /filter)
        try:
            d = s["date"]
            if d not in cache:
                cache[d] = _day_ctx(ts, c, d, tf, periods)
            ser, cnt, em = cache[d]
            idx = 1440 // tf + (s["hour"] * 60 + s["minute"]) // tf - 1
            if 0 <= idx < len(ser) and ser[idx] is not None:
                s["px"], s["cnt"], s["ema"] = ser[idx], cnt[idx], {p: em[p][idx] for p in periods}
            else:
                s["px"], s["cnt"], s["ema"] = None, 0, {}
        except Exception:  # noqa
            log.exception("engine13 ema context failed")
            s["px"], s["cnt"], s["ema"] = None, 0, {}
    return out


# ---------------------------------------------------------------- 4) texts
def _mtg_join(st):
    return "+".join(MTG_LB[k] for k in sorted(st["mtg"]))
def head_line(st):
    return "%d মিনিট • %s • %s • পরপর %d লস (%s) • কমপক্ষে %d সিগন্যাল, ১০০%% উইন" % (
        st["tf"], months_label(st["months"]), _mtg_join(st), st["n"], MODE_LB[st["mode"]], st["minsig"])
def list_texts(d, results, st):
    """The back-test list of ONE test day (same layout as /engine11, so it can also be pasted into /otctest)."""
    entries, spans, short_notes = [], [], []
    total = 0
    for r in results:
        x = r["lists"].get(d)
        entries.append((pairs.pretty(r["pair"]), None if x is None else x["rules"]))
        if x:
            total += len(x["rules"])
            spans += [x["first"], x["last"]]
            if x["short"]:
                short_notes.append("⚠️ %s: %d দিনের ডেটা চেয়েছিলে, ডাটাবেজে পেয়েছি প্রায় %d দিন — যেটুকু পেয়েছি সেটুকু দিয়েই ব্যাক-টেস্ট চালিয়েছি।" % (
                    pairs.pretty(r["pair"]), x["short"]["want"], x["short"]["got"]))
    head = "🎯 ইঞ্জিন ১৩ তালিকা — %s-এর টেস্টের জন্য\n%s\n📆 ব্যাক-টেস্ট হয়েছে: %s\nমোট রাডার স্লট: %d" % (
        d.strftime("%d %b %Y"), head_line(st),
        ("%s → %s (টেস্টের আগের দিন পর্যন্ত)" % (min(spans).isoformat(), max(spans).isoformat())) if spans else "ডেটা পাওয়া যায়নি", total)
    if short_notes:
        head += "\n\n" + "\n".join(short_notes)
    texts = _compose(head, entries)
    if any(pairs.is_otc(r["pair"]) for r in results):
        texts[0] = "🧪 OTC মার্কেট (ডাটাবেজ থেকে)\n\n" + texts[0]
    return texts
def _row_icon(s):
    if s["res"] != 1:
        return "❌"
    return "✅" + SUP.get(s["lvl"], "") if s["lvl"] else "✅"
def _lv_txt(lv):
    return " • ".join("%s %d" % ("সরাসরি উইন" if k == 0 else "MTG%d উইন" % k, lv[k]) for k in sorted(lv)) or "—"


def combo_txt(r):
    return "%d/%d • %s" % (r["nf"], r["ns"], FMODE_LB[r["fm"]])


def _pct(w, n):
    return "%.1f%%" % (100.0 * w / n) if n else "—"


def _t_line(t):
    s = "%d সিগন্যাল • ✅ %d • ❌ %d • 🎯 %s" % (t["n"], t["w"], t["l"], _pct(t["w"], t["n"]))
    return s + (" • P/L %+.1f" % t["pl"] if t["n"] else "")


def _pack(lines):
    """plain lines (escaped here) -> messages <= 3900 chars, sent with parse_mode HTML."""
    out, cur = [], ""
    for ln in lines:
        ln = html.escape(ln, quote=False)
        if cur and len(cur) + len(ln) + 1 > 3900:
            out.append(cur)
            cur = ""
        cur += ln + "\n"
    if cur.strip():
        out.append(cur)
    return out


def _day_messages(sigs, st, d, width, note=""):
    """ONE test day -> HTML messages: date heading (+ the filter used) + that day's signal rows. Long days are split."""
    rows = ["M%d %s %02d:%02d %s %s" % (st["tf"], pairs.pretty(s["pair"]).ljust(width), s["hour"], s["minute"],
                                       ("BUY" if s["side"] == "G" else "PUT"), _row_icon(s))
            for s in sorted((x for x in sigs if x["date"] == d), key=lambda x: (x["hour"], x["minute"], x["pair"]))]
    head = "📅 - %s\n▬▬▬▬▬▬▬▬▬ • ▬▬▬▬▬▬▬▬▬\n" % d.strftime("%Y.%m.%d")
    if note:
        head += html.escape(note, quote=False) + "\n"
    parts, cur, size = [], [], 0
    for r in rows:
        if cur and size + len(r) + 1 > 3300:
            parts.append(cur)
            cur, size = [], 0
        cur.append(r)
        size += len(r) + 1
    if cur:
        parts.append(cur)
    out = []
    for i, part in enumerate(parts):
        h = head if len(parts) == 1 else head + "(%d/%d)\n" % (i + 1, len(parts))
        out.append(h + "<pre>%s</pre>" % html.escape("\n".join(part), quote=False))
    return out


def summary_texts(sigs, wf, info, st, dates, slots):
    """HTML messages, in order: 1) summary + per-day numbers + best combos  2) one message per day with the signals that survived the
    filter  3) the losses that are still left + notes."""
    otc = any(pairs.is_otc(p) for p in st["pairs"])
    base = tally(sigs)
    L = ["📊 ইঞ্জিন ১৩ %s রিপোর্ট (ইঞ্জিন ১২ + EMA ফিল্টার) • %s" % ("OTC" if otc else "রিয়েল", dates_label(dates)),
         "%d মিনিট • %s • %s • পরপর %d লস (%s)" % (st["tf"], months_label(st["months"]), _mtg_join(st), st["n"], MODE_LB[st["mode"]]),
         "%d পেয়ার • রাডার স্লট: %s" % (len(st["pairs"]), " • ".join("%s: %d" % (d.strftime("%d %b"), slots.get(d, 0)) for d in dates)),
         "📈 EMA গ্রিড: Fast %s × Slow %s • %s" % (",".join(map(str, st["fast"])), ",".join(map(str, st["slow"])),
                                                " + ".join(FMODE_LB[m] for m in ("with", "counter") if m in st["fmodes"])),
         "ℹ️ সিগন্যাল ইঞ্জিন ১২-এর মতোই। EMA কম্বো প্রতিদিন বাছা হয়েছে শুধু তার আগের দিনগুলোর ফলাফল দিয়ে।", ""]
    if base["n"] == 0:
        L.append("এই দিন(গুলো)তে কোনো সিগন্যাল হয়নি।")
    else:
        L += ["① ইঞ্জিন ১২ — ফিল্টার ছাড়া (সব দিন):", _t_line(base), "   " + _lv_txt(base["lv"])]
        used = [r for r in wf if r["combo"]]
        if not used:
            L += ["", "② ওয়াক-ফরওয়ার্ড টেস্ট হয়নি: " + ("একটা দিনে শেষ ফলাফল দেখার মতো আগের দিন নেই।" if len(dates) < 2 else
                                                         "আগের দিনগুলোতে কোনো EMA কম্বোতেই যথেষ্ট সিগন্যাল টেকেনি।") + " আরও দিন দাও।"]
        else:
            ob = tally([s for r in used for s in r["all"]])
            ok = tally([s for r in used for s in r["kept"]])
            rem = tally([s for r in used for s in r["removed"]])
            L += ["", "② ওয়াক-ফরওয়ার্ড (সত্যিকারের পরীক্ষা) — %d দিনে:" % len(used),
                  "ফিল্টার ছাড়া: " + _t_line(ob), "ফিল্টারসহ  : " + _t_line(ok),
                  "বাদ গেছে   : %d টা (✅ %d • ❌ %d)" % (rem["n"], rem["w"], rem["l"])]
            gain = (100.0 * ok["w"] / ok["n"] - 100.0 * ob["w"] / ob["n"]) if ok["n"] and ob["n"] else None
            if ok["n"] < 20 or gain is None:
                L.append("🧐 মূল্যায়ন: ফিল্টারের পরে সিগন্যাল খুব কম (%d), ফল ভরসাযোগ্য না। আরও দিন যোগ করো।" % ok["n"])
            elif gain >= 3 and ok["pl"] > ob["pl"]:
                L.append("🧐 মূল্যায়ন: ফিল্টারসহ রেশিও %+.1f পয়েন্ট বেশি এবং P/L-ও ভালো। তবে আরও দিনে যাচাই করো।" % gain)
            elif gain >= 3:
                L.append("🧐 মূল্যায়ন: রেশিও %+.1f পয়েন্ট বেড়েছে, কিন্তু সিগন্যাল কমে P/L ভালো হয়নি। সাবধান।" % gain)
            else:
                L.append("🧐 মূল্যায়ন: EMA ফিল্টার এই ডেটায় আসল সুবিধা দিচ্ছে না (%+.1f পয়েন্ট)।" % gain)
        L += ["", "দিন ধরে:"]
        for r in wf:
            b = tally(r["all"])
            nm = r["date"].strftime("%d %b")
            if not b["n"]:
                L.append("  %s: সিগন্যাল নেই" % nm)
            elif not r["combo"]:
                L.append("  %s: কম্বো বাছার মতো আগের ডেটা নেই • ছাড়া %d/%d (%s)" % (nm, b["w"], b["n"], _pct(b["w"], b["n"])))
            else:
                k = tally(r["kept"])
                L.append("  %s: %s • ছাড়া %d/%d (%s) → সহ %d/%d (%s)" % (
                    nm, combo_txt(r["combo"]), b["w"], b["n"], _pct(b["w"], b["n"]), k["w"], k["n"], _pct(k["w"], k["n"])))
        rows = research(sigs, st["fast"], st["slow"], st["fmodes"])
        L += ["", "🔬 সব দিন মিলিয়ে সেরা কম্বো (একই ডেটায় বাছা, তাই বাস্তবে কম হবে):"]
        if not rows:
            L.append("কোনো কম্বোতে যথেষ্ট সিগন্যাল টেকেনি।")
        for i, r in enumerate(rows[:3]):
            L.append("%s %s • সহ %d/%d (%s) ← ছাড়া %d/%d (%s)" % ("🥇🥈🥉"[i], combo_txt(r), r["k"]["w"], r["k"]["n"], _pct(r["k"]["w"], r["k"]["n"]),
                                                              r["b"]["w"], r["b"]["n"], _pct(r["b"]["w"], r["b"]["n"])))
        for fm in ("with", "counter"):
            r = next((x for x in rows if x["fm"] == fm), None)
            if r and fm in st["fmodes"] and len(st["fmodes"]) > 1:
                L.append("📌 %s সেরা: %d/%d → %s" % (FMODE_LB[fm], r["nf"], r["ns"], _pct(r["k"]["w"], r["k"]["n"])))
        L.append("⚠️ ওপরের ওয়াক-ফরওয়ার্ডটাই আসল পরীক্ষা। এই তালিকা শুধু আইডিয়া দেয়।")
    out = _pack(L)                                              # 1) summary
    if base["n"] and st["show_kept"]:                           # 2) filtered signals, one message per day
        width = max(len(pairs.pretty(s["pair"])) for s in sigs)
        for r in wf:
            if r["combo"] and r["kept"]:
                out += _day_messages(r["kept"], st, r["date"], width, "ফিল্টার: " + combo_txt(r["combo"]))
    tail = []                                                   # 3) losses that are still left + notes
    bad = [s for r in wf for s in r["kept"] if s["res"] == -1]
    if bad:
        tail.append("❌ ফিল্টারের পরেও যে লসগুলো থাকল:")
        for s in bad:
            tail.append("  %s %02d:%02d %s %s" % (s["date"].strftime("%d %b"), s["hour"], s["minute"], pairs.pretty(s["pair"]), "UP" if s["side"] == "G" else "DOWN"))
    notes = []
    if info["nolist"]:
        notes.append("⚠️ তালিকা বানানোর মতো ডেটা ছিল না: %d টা (পেয়ার/দিন), যেমন %s %s।" % (
            len(info["nolist"]), pairs.pretty(info["nolist"][0][0]), info["nolist"][0][1].strftime("%d %b")))
    if info["no_data"]:
        nd = sorted({d for _, d in info["no_data"]})
        notes.append("⚠️ টেস্টের দিনের ক্যান্ডেল নেই: %d টা (পেয়ার/দিন), তারিখ: %s।" % (len(info["no_data"]), ", ".join(d.strftime("%d %b") for d in nd)))
    if info["partial"]:
        notes.append("⚠️ অসম্পূর্ণ দিন: %d টা (পেয়ার/দিন), যেমন %s %s-এর ডেটা শেষ %s পর্যন্ত।" % (
            len(info["partial"]), pairs.pretty(info["partial"][0][0]), info["partial"][0][1].strftime("%d %b"), info["partial"][0][2]))
    if info["missing"]:
        notes.append("ℹ️ ক্যান্ডেল না থাকায় %d টা সিগন্যাল বাদ গেছে।" % info["missing"])
    if notes:
        tail += ([""] if tail else []) + notes
    return out + _pack(tail)


def make_csv(sigs, wf):
    dec = {}
    for r in wf:
        for s in r["kept"]:
            dec[id(s)] = ("kept", combo_txt(r["combo"]))
        for s in r["removed"]:
            dec[id(s)] = ("removed", combo_txt(r["combo"]))
    bio = io.StringIO()
    w = csv.writer(bio)
    w.writerow(["date", "time_bdt", "pair", "direction", "result", "win_level", "doji_win", "losses_before", "filter_decision", "filter_combo"])
    for s in sigs:
        dd, cb = dec.get(id(s), ("no_filter", ""))
        w.writerow([s["date"].isoformat(), "%02d:%02d" % (s["hour"], s["minute"]), pairs.pretty(s["pair"]), "UP" if s["side"] == "G" else "DOWN",
                    "WIN" if s["res"] == 1 else "LOSS", (("Non-MTG" if s["lvl"] == 0 else "MTG%d" % s["lvl"]) if s["res"] == 1 else ""), int(s["doji"]),
                    s["streak"], dd, cb])
    return ("\ufeff" + bio.getvalue()).encode("utf-8")


# ---------------------------------------------------------------- 5) settings screen
def e13_text():
    ps = ", ".join(pairs.pretty(p) for p in E["pairs"]) or "— (বাছাই করোনি)"
    mtg = " + ".join(MTG_LB[v] for v in sorted(E["mtg"])) or "— (বাছাই করোনি)"
    files = " + ".join(x for x, f in (("রাডার তালিকা", E["show_list"]), ("ফিল্টারের পরের সিগন্যাল", E["show_kept"]), ("CSV", E["csv"])) if f) or "শুধু রিপোর্ট"
    lines = [
        "🧪 ইঞ্জিন ১৩ — ইঞ্জিন ১২ + EMA ফিল্টার", "",
        "ইঞ্জিন ১২-এর মতোই প্রতিটা টেস্ট-দিনের সিগন্যাল বানায় ও রিপ্লে করে। তারপর EMA (Fast/Slow) গ্রিডের সব কম্বো পরীক্ষা করে দেখে কোনটা "
        "ট্রেন্ড দিয়ে সিগন্যাল সবচেয়ে ভালো ছাঁকে। প্রতিদিনের কম্বো বাছা হয় শুধু আগের দিনগুলোর ফলাফল দিয়ে, তাই ফলটা সৎ। "
        "ভালো ফলের জন্য ৭+ দিন দাও (কমপক্ষে ২ দিন লাগবে)।", "",
        "⚙️ সেটিং:",
        "• টেস্টের দিন: %s%s" % (dates_label(), (" (%d দিন)" % len(test_dates())) if len(test_dates()) > 1 else ""),
        "• টাইম ফ্রেম: %d মিনিট" % E["tf"],
        "• ব্যাক-টেস্ট সময়কাল: %s (প্রতি টেস্ট-দিনের আগের দিন পর্যন্ত)" % months_label(E["months"]),
        "• MTG: %s" % mtg,
        "• ব্যাক-টু-ব্যাক লস: %d টা (%s)" % (E["n"], MODE_LB[E["mode"]]),
        "• ন্যূনতম সিগন্যাল: %d" % E["minsig"],
        "• EMA গ্রিড: Fast %s × Slow %s" % (",".join(map(str, E["fast"])), ",".join(map(str, E["slow"]))),
        "• ট্রেন্ড মোড: %s" % (" + ".join(FMODE_LB[m] for m in ("with", "counter") if m in E["fmodes"]) or "— (বাছাই করোনি)"),
        "• সাথে পাঠাবে: %s" % files,
        "• পেয়ার: %s" % ps, "",
        "🧪 মার্কেট: OTC — পুরো ২৪ ঘণ্টা (০০:০০–২৩:৫৯)" if E["mkt"] == "otc" else "🌍 মার্কেট: রিয়েল — দুপুর ১২:০০ থেকে ২৩:৫৯ (BDT)",
        "(একাধিক দিন: ✏️ তারিখ চেপে 14-18 অথবা 14,15,16,17,18 লেখো • একসাথে সর্বোচ্চ %d দিন)" % MAX_DAYS]
    if E["running"]:
        lines += ["", "⏳ একটা টেস্ট চলছে।"]
    return "\n".join(lines)


def e13_kb():
    if E["page"] == "pairs":
        rows, cur = [], []
        for p in pairs.plist(E["mkt"]):
            cur.append(Btn(_tick(p in E["pairs"], pairs.pretty(p)), callback_data="e13:tp:%s" % p))
            if len(cur) == 3:
                rows.append(cur)
                cur = []
        if cur:
            rows.append(cur)
        rows.append([Btn("সব পেয়ার", callback_data="e13:pall:0"), Btn("পরিষ্কার", callback_data="e13:pclr:0"), Btn("⬅️ ফিরে", callback_data="e13:pg:main")])
        return Markup(rows)
    custom = E["days"] == "custom"
    default_grid = E["fast"] == list(FAST_GRID) and E["slow"] == list(SLOW_GRID)
    return Markup([
        [Btn(_tick(E["mkt"] == v, lb), callback_data="e13:mk:%s" % v) for v, lb in (("real", "🌍 রিয়েল মার্কেট"), ("otc", "🧪 OTC মার্কেট"))],
        [Btn(_tick(E["days"] == "y", "গতকাল"), callback_data="e13:dy:y"), Btn(_tick(E["days"] == "t", "আজ"), callback_data="e13:dy:t"),
         Btn(_tick(custom, ("✏️ " + dates_label()) if custom and E["dates"] else "✏️ তারিখ"), callback_data="e13:dy:custom")],
        [Btn(_tick(E["tf"] == v, "%d মিনিট" % v), callback_data="e13:tf:%d" % v) for v in TF_PRESETS]
        + [Btn(_tick(E["tf"] not in TF_PRESETS, "✏️ %d মিনিট" % E["tf"] if E["tf"] not in TF_PRESETS else "✏️"), callback_data="e13:tf:custom")],
        [Btn(_tick(E["months"] == v, "%dM" % v), callback_data="e13:mo:%d" % v) for v in MONTH_PRESETS]
        + [Btn(_tick(E["months"] not in MONTH_PRESETS, "✏️ " + months_label(E["months"]) if E["months"] not in MONTH_PRESETS else "✏️"), callback_data="e13:mo:custom")],
        [Btn(_tick(v in E["mtg"], lb), callback_data="e13:mtg:%d" % v) for v, lb in MTG_LB.items()],
        [Btn(_tick(E["n"] == v, "লস %d" % v), callback_data="e13:n:%d" % v) for v in range(1, 7)],
        [Btn(_tick(E["n"] > 6, "✏️ লস %d (কাস্টম)" % E["n"] if E["n"] > 6 else "✏️ লস কাস্টম"), callback_data="e13:n:custom")],
        [Btn(_tick(E["mode"] == v, lb), callback_data="e13:md:%s" % v) for v, lb in MODE_LB.items()],
        [Btn(_tick(E["minsig"] == v, "সি. ≥%d" % v), callback_data="e13:ms:%d" % v) for v in MINSIG_PRESETS]
        + [Btn(_tick(E["minsig"] not in MINSIG_PRESETS, "✏️ সি. ≥%d" % E["minsig"] if E["minsig"] not in MINSIG_PRESETS else "✏️"), callback_data="e13:ms:custom")],
        [Btn(_tick(default_grid, "📈 EMA ডিফল্ট গ্রিড"), callback_data="e13:gr:def"), Btn(_tick(not default_grid, "✏️ EMA কাস্টম"), callback_data="e13:gr:custom")],
        [Btn(_tick(m in E["fmodes"], lb), callback_data="e13:fm:%s" % m) for m, lb in FMODE_LB.items()],
        [Btn(_tick(E["show_list"], "📋 রাডার তালিকা"), callback_data="e13:sl:0"), Btn(_tick(E["show_kept"], "📌 ফিল্টারের পরের সিগন্যাল"), callback_data="e13:sk:0"),
         Btn(_tick(E["csv"], "📊 CSV"), callback_data="e13:csv:0")],
        [Btn("💱 পেয়ার (%d)" % len(E["pairs"]), callback_data="e13:pg:pairs")],
        [Btn("⏹ বন্ধ করো", callback_data="e13:stop:0") if E["running"] else Btn("▶️ চালাও", callback_data="e13:run:0")]])


# ---------------------------------------------------------------- 6) the job
async def _job(chat_id, st, dates):
    send = D["send"]
    E["running"], E["stop"] = True, False
    kb_stop = Markup([[Btn("⏹ বন্ধ করো", callback_data="e13:stop:0")]])
    prog = None
    try:
        prog = await send(chat_id, "⏳ ইঞ্জিন ১৩: %s — তালিকা বানাচ্ছি, রিপ্লে ও EMA ফিল্টার পরীক্ষা করছি…" % dates_label(dates), reply_markup=kb_stop)
        results = []
        async with D["lock"]:
            for i, p in enumerate(st["pairs"]):
                if E["stop"]:
                    break
                results.append(await asyncio.to_thread(compute_pair, p, dates, st))
                if prog:
                    try:
                        await prog.edit_text("⏳ ইঞ্জিন ১৩: %d/%d পেয়ার শেষ" % (i + 1, len(st["pairs"])), reply_markup=kb_stop)
                    except Exception:  # noqa
                        pass
        if E["stop"]:
            return await send(chat_id, "⏹ ইঞ্জিন ১৩ বন্ধ করা হয়েছে।")
        sigs = sorted((s for r in results for s in r["sigs"]), key=lambda x: (x["ts"], x["pair"], x["side"]))
        info = dict(nolist=[(r["pair"], d) for r in results for d in dates if r["lists"].get(d) is None],
                    no_data=[(r["pair"], d) for r in results for d in r["no_data"]],
                    partial=[(r["pair"], d, h) for r in results for d, h in r["partial"]],
                    missing=sum(r["missing"] for r in results))
        slots = {d: sum(len(r["lists"][d]["rules"]) for r in results if r["lists"].get(d)) for d in dates}
        wf = await asyncio.to_thread(walk_forward, sigs, dates, st)
        if st["show_list"]:
            for d in dates:
                for t in list_texts(d, results, st):
                    if E["stop"]:
                        break
                    await send(chat_id, t[:4000])
        texts = await asyncio.to_thread(summary_texts, sigs, wf, info, st, dates, slots)
        for t in texts:
            if E["stop"]:
                break
            await send(chat_id, t, parse_mode="HTML")
        if E["stop"]:
            await send(chat_id, "⏹ ইঞ্জিন ১৩ বন্ধ করা হয়েছে।")
        elif sigs and st["csv"]:
            bio = io.BytesIO(make_csv(sigs, wf))
            bio.name = "engine13_%s.csv" % dt.datetime.now(BDT).strftime("%Y%m%d_%H%M")
            try:
                await D["bot"]().send_document(chat_id, bio, caption="প্রতিটা সিগন্যালের তারিখ, সময়, রেজাল্ট ও ফিল্টারের সিদ্ধান্ত")
            except Exception:  # noqa
                log.exception("engine13 csv failed")
        if prog:
            try:
                await prog.delete()
            except Exception:  # noqa
                pass
    except Exception as e:  # noqa
        log.exception("engine13 failed")
        await send(chat_id, "❌ ইঞ্জিন ১৩ হয়নি: %s" % e)
    finally:
        E["running"] = False


# ---------------------------------------------------------------- 7) Telegram glue
def _release_e12():
    """If /engine12 happens to wait for a typed answer, let go of it (best effort; engine 13 works without engine 12)."""
    m = sys.modules.get("engine12")
    try:
        if m is not None:
            m.E["wait"] = False
    except Exception:  # noqa
        pass


async def cmd_engine13(update, ctx):
    if not D["allowed"](update):
        return await D["deny"](update)
    only = [p for p in (pairs.norm(t) for a in (ctx.args or []) for t in a.replace(",", " ").split()) if p]
    if only:
        E["pairs"] = list(dict.fromkeys(only))
        E["mkt"] = "otc" if pairs.is_otc(only[0]) else "real"
    E["page"], E["wait"] = "main", False
    _release_e12()
    await update.effective_message.reply_text(style.fancy(e13_text()), reply_markup=e13_kb(), parse_mode="HTML")


def _ask(key):
    return {"tf": "✏️ টাইম ফ্রেম কত মিনিট চাও লেখো (যেমন: 2, 3, 4, 5, 10, 15)।\nসংখ্যাটা ৬০-কে ভাগ করতে পারে এমন হতে হবে।",
            "months": "✏️ কত দিনের ব্যাক-টেস্ট চাও লেখো:\n• মাস: 7 অথবা 7M\n• দিন: 45D\n• বছর: 1Y\n(সর্বোচ্চ %d দিন)" % E3_MAX_DAYS,
            "minsig": "✏️ রাডারে ঢুকতে একটা মিনিটে কমপক্ষে কয়টা সিগন্যাল লাগবে লেখো (১ থেকে ৫০)।",
            "nloss": "✏️ পরপর (ব্যাক-টু-ব্যাক) কয়টা লস হলে সিগন্যাল নিতে চাও লেখো (১ থেকে ২০)।\nযেমন: 7",
            "grid": "✏️ EMA গ্রিড লেখো: Fast-এর সংখ্যাগুলো / Slow-এর সংখ্যাগুলো\nযেমন: 5,13,21 / 34,50,100\nশুধু একটা জোড়া দেখতে: 13/50\n(প্রতিটা ২ থেকে %d; প্রতি দিকে সর্বোচ্চ ৮টা)" % MAX_PERIOD,
            "dates": "📅 কোন কোন দিন টেস্ট করবে লেখো:\n• একটা দিন: 14\n• থেকে-পর্যন্ত: 14-18 (১৪ থেকে ১৮ তারিখ সবগুলো)\n• আলাদা আলাদা: 14,15,16,17,18 (কমা বা স্পেস দিয়ে)\n• অন্য মাস: 14/10 অথবা 2026-10-14\n(আজ বা আগের তারিখ; একসাথে সর্বোচ্চ %d দিন)" % MAX_DAYS}[key]


async def on_cb(update, ctx):
    q = update.callback_query
    if not D["allowed"](update):
        return await q.answer("অনুমতি নেই", show_alert=True)
    parts = q.data.split(":")
    key, val = parts[1], parts[2]
    if key == "stop":
        if not E["running"]:
            return await q.answer("কিছু চলছে না")
        E["stop"] = True
        return await q.answer("বন্ধ করছি…")
    if key == "run":
        if E["running"]:
            return await q.answer("একটা টেস্ট চলছে। শেষ হলে চাপো।", show_alert=True)
        if not E["pairs"]:
            return await q.answer("আগে অন্তত একটা পেয়ার বাছো", show_alert=True)
        miss = data_missing(E["pairs"])
        if miss:
            return await q.answer(miss, show_alert=True)
        if not E["mtg"]:
            return await q.answer("অন্তত একটা MTG অপশন বাছো", show_alert=True)
        if not E["fmodes"]:
            return await q.answer("অন্তত একটা ট্রেন্ড মোড বাছো (সাথে / উল্টো)", show_alert=True)
        if E["days"] == "custom" and not E["dates"]:
            return await q.answer("✏️ তারিখ বাছা আছে কিন্তু তারিখ লেখোনি। ✏️ চেপে লেখো, নয়তো গতকাল/আজ বাছো।", show_alert=True)
        dates = test_dates()
        today = dt.datetime.now(BDT).date()
        if any(d > today for d in dates):
            return await q.answer("ভবিষ্যতের তারিখ। আজ বা আগের তারিখ দাও।", show_alert=True)
        if len(dates) > MAX_DAYS:
            return await q.answer("একসাথে সর্বোচ্চ %d দিন। কম দিন দাও।" % MAX_DAYS, show_alert=True)
        if D["lock"].locked():
            return await q.answer("একটা স্ক্যান চলছে, শেষ হলে চাপো।", show_alert=True)
        await q.answer("শুরু হচ্ছে…")
        st = dict(E)
        st["mtg"], st["pairs"] = sorted(E["mtg"]), list(E["pairs"])
        st["fast"], st["slow"], st["fmodes"] = list(E["fast"]), list(E["slow"]), list(E["fmodes"])
        E["wait"] = False
        E["running"] = True                      # set now, so a quick second tap cannot start a second job
        ctx.application.bot_data["engine13_job"] = asyncio.create_task(_job(q.message.chat_id, st, dates))
        return
    await q.answer()
    if key == "mk":
        E["mkt"] = "otc" if val == "otc" else "real"
        E["pairs"] = []
        E["page"] = "main"
        E["wait"] = False
    elif key == "pg":
        E["page"] = val
    elif key == "dy" and val == "custom":
        E["days"], E["wait"], E["wait_at"] = "custom", "dates", time.time()
        await q.message.reply_text(_ask("dates"))
        try:
            await q.edit_message_text(style.fancy(e13_text()), reply_markup=e13_kb(), parse_mode="HTML")
        except BadRequest:
            pass
        return
    elif key == "dy":
        E["days"], E["wait"] = val, False
    elif key == "gr" and val == "custom":
        E["wait"], E["wait_at"] = "grid", time.time()
        await q.message.reply_text(_ask("grid"))
        return
    elif key == "gr":
        E["fast"], E["slow"], E["wait"] = list(FAST_GRID), list(SLOW_GRID), False
    elif key == "fm":
        cur = list(E["fmodes"])
        E["fmodes"] = [x for x in cur if x != val] if val in cur else [x for x in ("with", "counter") if x in cur or x == val]
    elif key in ("tf", "mo", "ms", "n") and val == "custom":
        E["wait"], E["wait_at"] = {"tf": "tf", "mo": "months", "ms": "minsig", "n": "nloss"}[key], time.time()
        await q.message.reply_text(_ask(E["wait"]))
        return
    elif key == "tf":
        E["wait"], E["tf"] = False, int(val)
    elif key == "mo":
        E["wait"], E["months"] = False, int(val)
    elif key == "ms":
        E["wait"], E["minsig"] = False, int(val)
    elif key == "n":
        E["wait"], E["n"] = False, int(val)
    elif key == "md":
        E["mode"] = val
    elif key == "mtg":
        v = int(val)
        E["mtg"] = [x for x in E["mtg"] if x != v] if v in E["mtg"] else sorted(E["mtg"] + [v])
    elif key == "sl":
        E["show_list"] = not E["show_list"]
    elif key == "sk":
        E["show_kept"] = not E["show_kept"]
    elif key == "csv":
        E["csv"] = not E["csv"]
    elif key == "tp":
        E["pairs"] = [p for p in E["pairs"] if p != val] if val in E["pairs"] else E["pairs"] + [val]
    elif key == "pall":
        E["pairs"] = list(pairs.plist(E["mkt"]))
    elif key == "pclr":
        E["pairs"] = []
    try:
        await q.edit_message_text(style.fancy(e13_text()), reply_markup=e13_kb(), parse_mode="HTML")
    except BadRequest:
        pass


async def on_text(update, ctx):
    """Group -5 handler (the first text handler to run; /otctest uses -1, /engine12 uses -2): eats a message only while a ✏️ prompt of /engine13 waits for its answer, otherwise lets everything else run."""
    w = E["wait"]
    if not w:
        return
    if time.time() - E["wait_at"] > WAIT_SECS:
        E["wait"] = False
        return
    if not D["allowed"](update):
        return
    msg = update.effective_message
    txt = _digits(msg.text or "").strip()
    if w == "dates":
        ds = parse_dates(txt, dt.datetime.now(BDT).date())
        if not ds:
            await msg.reply_text("তারিখ বুঝিনি। এভাবে লেখো: 17 / 17,18,19 / 17-19 / 17/11 / 2026-11-17 / গতকাল")
            raise ApplicationHandlerStop
        if any(d > dt.datetime.now(BDT).date() for d in ds):
            await msg.reply_text("ভবিষ্যতের তারিখ দিয়েছ। আজ বা আগের তারিখ দাও।")
            raise ApplicationHandlerStop
        if len(ds) > MAX_DAYS:
            await msg.reply_text("একসাথে সর্বোচ্চ %d দিন। কম দিন দাও।" % MAX_DAYS)
            raise ApplicationHandlerStop
        E["dates"], E["days"] = [d.isoformat() for d in ds], "custom"
    elif w == "grid":
        g = parse_grid(txt)
        if not g:
            await msg.reply_text("বুঝিনি। এভাবে লেখো: 5,13,21 / 34,50,100 (প্রতিটা ২–%d, প্রতি দিকে সর্বোচ্চ ৮টা, Fast Slow-এর চেয়ে ছোট থাকতে হবে)" % MAX_PERIOD)
            raise ApplicationHandlerStop
        E["fast"], E["slow"] = g
    elif w == "tf":
        t = txt.lower().replace("মিনিট", "").replace("min", "").replace("m", "").strip()
        if not t.isdigit() or not (1 <= int(t) <= 30) or 60 % int(t):
            await msg.reply_text("বুঝিনি। এমন একটা সংখ্যা লেখো যেটা ৬০-কে ভাগ করে: 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30")
            raise ApplicationHandlerStop
        E["tf"] = int(t)
    elif w == "minsig":
        if not txt.isdigit() or not (1 <= int(txt) <= 50):
            await msg.reply_text("বুঝিনি। ১ থেকে ৫০-এর মধ্যে একটা সংখ্যা লেখো।")
            raise ApplicationHandlerStop
        E["minsig"] = int(txt)
    elif w == "nloss":
        t = txt.lower().replace("লস", "").replace("loss", "").strip()
        if not t.isdigit() or not (1 <= int(t) <= 20):
            await msg.reply_text("বুঝিনি। ১ থেকে ২০-এর মধ্যে একটা সংখ্যা লেখো (যেমন: 7)।")
            raise ApplicationHandlerStop
        E["n"] = int(t)
    else:
        months = parse_period(txt)
        if months is None:
            await msg.reply_text("বুঝিনি। এভাবে লেখো: 7 / 7M / 45D / 1Y (১–%d দিনের মধ্যে)" % E3_MAX_DAYS)
            raise ApplicationHandlerStop
        E["months"] = months
    E["wait"], E["page"] = False, "main"
    await msg.reply_text(style.fancy(e13_text()), reply_markup=e13_kb(), parse_mode="HTML")
    raise ApplicationHandlerStop


async def _clear_wait(update, ctx):
    """Another engine's command / button was used: forget a pending ✏️ prompt of /engine13 (does not stop the other handler)."""
    E["wait"] = False


def register(app, send, allowed, deny, lock, bot_getter):
    """Called once from bot.py main(): own command (/engine13), callbacks and text handler. Own handler groups (-5 for text, -4 for the clear-wait
    handlers) that no other engine uses: PTB runs only ONE handler per group, so a shared group would silently swallow our text."""
    D.update(send=send, allowed=allowed, deny=deny, lock=lock, bot=bot_getter)
    app.add_handler(CommandHandler("engine13", cmd_engine13))
    app.add_handler(CallbackQueryHandler(on_cb, pattern=r"^e13:"))
    app.add_handler(CommandHandler(["engine3", "engine4", "engine6", "engine8", "engine9", "engine10", "engine11", "engine12", "otctest", "realtest"], _clear_wait), group=-4)
    app.add_handler(CallbackQueryHandler(_clear_wait, pattern=r"^(e3|e4|e6|e8|e9|e10|e11|e12|ot|rt):"), group=-4)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text), group=-5)


COMMAND = BotCommand("engine13", "ইঞ্জিন ১৩: ইঞ্জিন ১২ + EMA ফিল্টার (সেরা EMA সেটিং খোঁজা)")

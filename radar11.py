"""Engine 11 — Signal Radar 11 = Engine 10 + a choice of WHERE the back-test period ends (own module, Engine 10 is untouched).

New in Engine 11: the back-test of the last N days can end at (a) the last candle day stored in the Neon database (exactly like Engine 10)
or (b) a custom date typed by the user (N days counted back from that date, that date included). Everything else is the same code and
the same rules as Engine 10. Engine 11 has its OWN radar list / on-off switch / alerts (kv keys radar11_*), so it never overwrites Engine 10.

Engine 10 text follows:

Engine 10 — Signal Radar (live alerts built from the Engine 9 back-test).

1) build():      for the chosen pairs run the Engine 9 back-test (Engine 8's lowest-B2B minutes -> Engine 4's "signal after N
                 back-to-back losses"). A slot (pair, colour, minute) goes on the radar list only when its win ratio is 100%
                 with at least `minsig` signals. If several MTG levels were chosen, the lowest MTG that qualifies is kept.
2) alert_tick(): ~LEAD seconds before every minute, for each radar slot that starts now, today's real candles are read from
                 OANDA (Neon is NOT touched) and today's hours 12:00.. of that minute are replayed with the SAME code as the
                 back-test (engine4.outcomes). If the losing streak is exactly N (or N+ in "N or more" mode) -> signal is sent
                 (green slot = UP, red slot = DOWN).
3) result_tick(): after the candles close, WIN / LOSS is sent with today's tally.
Settings, the radar list and the alerts live in memory and are saved to the results DB (kv) only when something changes."""
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
# EMBEDDED COPY of radar_report11.py (identical code, private to this file)
# ==========================================================================
def _build_radar_report11():
    def _g():
        return _ns

    """Engine 11 report files (copy of radar_report.py, only the titles differ) (PDF + CSV) built from the radar build result. Pure reportlab / csv, no Telegram, no database.
    PDF text is English (Bengali text needs a shaping library that most servers do not have); the Telegram text stays Bengali."""
    import csv
    import io

    e3 = engine3
    pass  # engine9 is embedded below
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
                                title="Signal Radar (Engine 11)")

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
        story = [Paragraph("Signal Radar - Engine 11 (back-test details)", h1)]
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


    _ns = dict(locals())
    return types.SimpleNamespace(**{k: v for k, v in _ns.items() if not k.startswith("__")})


radar_report11 = _build_radar_report11()


import asyncio
import datetime as dt
import json
import logging
import time

import numpy as np

pass  # engine3 is embedded below
pass  # engine4 is embedded below
pass  # engine9 is embedded below
import live
import oanda
import pairs
import resultsdb as rdb
import source

log = logging.getLogger("radar11")
LEAD, RES_SEC, STALE, GRACE = live.LEAD, live.RES_SEC, live.STALE, live.GRACE
REFRESH_HM = (11, 30)                  # BDT: the list is rebuilt every day from this time (before the 12:00 trading window)
HOURS = engine3.HOURS
UP, DOWN = "UP (Call) 📈", "DOWN (Put) 📉"

DIAG = {"tick_at": 0.0, "last_alert": None, "skipped": "", "requests": 0, "res_at": 0.0, "refresh_err": "", "try_at": 0.0}
_cfg = {"on": False, "chat": None}
_state = {"rules": [], "meta": {}, "built_at": 0}
_alerts = {}      # key -> dict(key, slot_ts, pair, side, mtg, tf, state, step)


# ---------------------------------------------------------------- persistence
def _save(key, obj):
    if rdb.enabled():
        rdb.kv_set(key, json.dumps(obj, ensure_ascii=False))


def _load(key):
    if not rdb.enabled():
        return None
    v = rdb.kv_get(key)
    return json.loads(v) if v else None


async def load_state():
    cfg = await asyncio.to_thread(_load, "radar11_cfg")
    if cfg:
        _cfg.update(on=bool(cfg.get("on")), chat=cfg.get("chat"))
    st = await asyncio.to_thread(_load, "radar11_rules")
    if st:
        _state.update(rules=st.get("rules", []), meta=st.get("meta", {}), built_at=st.get("built_at", 0))
    al = await asyncio.to_thread(_load, "radar11_alerts") or []
    lim = live.day_start(time.time()) - 86400
    for a in al:
        if a["slot_ts"] >= lim:
            _alerts[a["key"]] = a


def cfg():
    return _cfg


async def set_cfg(**kw):
    _cfg.update(kw)
    await asyncio.to_thread(_save, "radar11_cfg", _cfg)


def state():
    return _state


async def set_rules(rules, meta):
    _state.update(rules=rules, meta=meta, built_at=int(time.time()))
    await asyncio.to_thread(_save, "radar11_rules", _state)


def _save_alerts():
    lim = live.day_start(time.time()) - 86400
    _save("radar11_alerts", [a for a in _alerts.values() if a["slot_ts"] >= lim])


def tally(since_ts):
    w = l = u = o = 0
    for a in _alerts.values():
        if a["slot_ts"] >= since_ts:
            w += a["state"] == "win"
            l += a["state"] == "loss"
            u += a["state"] == "unknown"
            o += a["state"] == "open"
    return w, l, u, o


# ---------------------------------------------------------------- 1) build the radar list
def end_bounds(st, days, now=None):
    """-> (from_ts, to_ts) UTC window of M1 candles to load for the back-test.
    src 'db' (default, same as Engine 10): everything up to now -> the last N days end at the last candle day of the database.
    src 'custom': candles up to the END of the chosen BDT date (UTC+6) -> the last N days end at that date (that date included)."""
    now = int(now or time.time())
    if st.get("src") == "custom" and st.get("end"):
        d = dt.date.fromisoformat(st["end"])
        next_midnight_local = ((d - dt.date(1970, 1, 1)).days + 1) * 86400
        hi = next_midnight_local - engine3.OFF            # 00:00 BDT of the next day, as UTC
        return hi - (days + 10) * 86400, hi
    return now - (days + 10) * 86400, now + 86400


def span_of(per):
    """Build result -> ('2026-08-14', '2026-11-13') first / last trading day really used (None when no pair had data)."""
    ds = []
    for e in per:
        if e:
            r = e["res"]
            ds.append(engine3.day_date(r["d0"], r["days"][0]))
            ds.append(engine3.day_date(r["d0"], r["days"][-1]))
    return (min(ds).isoformat(), max(ds).isoformat()) if ds else None


def short_notes(per, st):
    """-> list of lines for the pairs whose database holds fewer days than the chosen period (they ran on the days that exist)."""
    out = []
    for p, e in zip(st["pairs"], per):
        if e and e.get("short"):
            x = e["short"]
            out.append("⚠️ %s: %d দিনের ডেটা চেয়েছিলে, ডাটাবেজে পেয়েছি প্রায় %d দিন (%s → %s) — যেটুকু পেয়েছি সেটুকু দিয়েই ব্যাক-টেস্ট চালিয়েছি।" % (
                pairs.pretty(p), x["want"], x["got"], x["first"], x["last"]))
    return out


def src_label(meta):
    if meta.get("src") == "custom" and meta.get("end"):
        return "কাস্টম তারিখ %s থেকে বিগত" % meta["end"]
    return "ডাটাবেজের শেষ দিন থেকে বিগত"


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


async def build(st, progress=None):
    """st: tf, months, mtg (list), n, mode ('exact'|'atleast'), minsig, pairs. Caller holds the scan lock.
    -> list of dict(pair, days, rules) or None entries (pair without data) in the order of st['pairs']."""
    days = engine3.period_days(st["months"])
    plist = list(st["pairs"])
    gate = asyncio.Semaphore(3)

    async def one(pair):
        async with gate:
            lo, hi = end_bounds(st, days)
            ts, o, c = await asyncio.to_thread(source.load_oc, pair, lo, hi)
            res = await asyncio.to_thread(engine9.analyze, pair, ts, o, c, st["tf"], st["months"], sorted(st["mtg"]),
                                          st["n"], "day", st["mode"] == "atleast")
            first_ts = int(ts[0]) if len(ts) else None
            del ts, o, c
            if res is None:
                return None
            # Engine 11: the database may hold fewer days than asked for -> the back-test simply runs on what exists; remember it to tell the user
            short = None
            if first_ts is not None:
                first_have = int((first_ts + engine3.OFF) // 86400) - res["d0"]        # first day number that has candles
                first_want = int(res["days"][-1]) - days + 1
                if first_have - first_want > 3:                                          # > 3 days missing (weekend gaps are not counted)
                    short = dict(want=days, got=int(res["days"][-1]) - int(res["days"][0]) + 1,
                                 first=engine3.day_date(res["d0"], res["days"][0]).isoformat(),
                                 last=engine3.day_date(res["d0"], res["days"][-1]).isoformat())
            return dict(pair=pair, days=len(res["days"]), rules=rules_from(res, st["minsig"]), res=res, short=short)

    tasks = [asyncio.create_task(one(p)) for p in plist]
    out = []
    try:
        for i, tk in enumerate(tasks):
            out.append(await tk)
            if progress:
                await progress(i + 1, len(tasks))
    except BaseException:
        for tk in tasks:
            tk.cancel()
        raise
    return out


def _mt(k):
    return "নন-MTG" if k == 0 else "MTG%d" % k


def _mins(rs, multi):
    return " ".join(":%02d%s" % (x["minute"], ("·M%d" % x["mtg"]) if multi else "") for x in rs) or "—"


def _compose(head, entries):
    """entries: [(pair name, rules or None)] -> list of messages (<= 3900 chars). Per pair: name + green minutes + red minutes
    (no counts); the very last block merges the minutes of all pairs, in serial order, each minute once."""
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


def summary_text(per, st, labels=None):
    """Build result -> messages."""
    sp = span_of(per)
    head = "🎯 সিগন্যাল রাডার ১১ তালিকা\n%d মিনিট • %s • %s • পরপর %d লস (%s) • কমপক্ষে %d সিগন্যাল, ১০০%% উইন\n📆 %s %s\n%s\nমোট রাডার স্লট: %d" % (
        st["tf"], st["months_label"], "+".join(_mt(k) for k in st["mtg"]), st["n"], "N বা বেশি" if st["mode"] == "atleast" else "ঠিক N", st["minsig"],
        src_label(st), st["months_label"],
        ("ব্যাক-টেস্ট হয়েছে: %s → %s" % sp) if sp else "ব্যাক-টেস্টের মতো ডেটা পাওয়া যায়নি",
        sum(len(e["rules"]) for e in per if e))
    notes = short_notes(per, st)
    if notes:
        head += "\n\n" + "\n".join(notes)
    entries = [(pairs.pretty(p), None if e is None else e["rules"]) for p, e in zip(st["pairs"], per)]
    return _compose(head, entries)


def list_text(labels=None):
    rules, meta = _state["rules"], _state["meta"]
    if not rules:
        return ["রাডার ১১ তালিকা এখনও খালি। আগে 🔄 তালিকা বানাও চাপো।"]
    when = dt.datetime.fromtimestamp(_state["built_at"], pairs.BDT).strftime("%d %b %H:%M")
    sp = meta.get("span")
    head = "🎯 রাডার ১১ তালিকা (বানানো: %s)\n%d মিনিট • %s • %s • পরপর %d লস • কমপক্ষে %d সিগন্যাল\n📆 %s %s%s\nমোট রাডার স্লট: %d" % (
        when, meta.get("tf", 1), meta.get("months_label", ""), "+".join(_mt(k) for k in meta.get("mtg", [])), meta.get("n", 0),
        meta.get("minsig", 0), src_label(meta), meta.get("months_label", ""), (" (%s → %s)" % tuple(sp)) if sp else "", len(rules))
    if meta.get("short"):
        head += "\n\n" + "\n".join(meta["short"])
    order = list(meta.get("pairs") or [])
    for x in rules:
        if x["pair"] not in order:
            order.append(x["pair"])
    entries = [(pairs.pretty(p), [x for x in rules if x["pair"] == p]) for p in order]
    return _compose(head, entries)


# ---------------------------------------------------------------- 2) live check (same code as the back-test)
def streak_info(ts, o, c, r, hour):
    """Today's M1 candles (ts, o, c) -> (due, streak, loss_hours) for rule r at trading hour `hour` (BDT).
    due == the back-test would take a signal on this slot (exact N, or N or more)."""
    tf = r["tf"]
    lt, d = engine3.to_tf(ts, o, c, tf)
    if len(lt) == 0:
        return None
    _, dirm, val = engine3.build_grid(lt, d, tf)
    out, _, _ = engine4.outcomes(dirm, val, tf, r["minute"], np.array([0]), r["side"], r["mtg"])
    streak, hrs = 0, []
    for j in range(max(0, hour - HOURS[0])):
        v = int(out[0, j])
        if v == 0:
            continue                      # no candle -> that trade is skipped, same as in the back-test
        if v == -1:
            streak += 1
            hrs.append(HOURS[j])
        else:
            streak, hrs = 0, []
    due = streak >= r["n"] if r["atleast"] else streak == r["n"]
    return due, streak, hrs


def judge(ts, o, c, slot, k, tf, side, now):
    """Candles from the slot on -> (state 'win'|'loss'|'unknown'|None, step, wait). None + wait=True -> check again later."""
    lt, d = engine3.to_tf(ts, o, c, tf) if len(ts) else (np.zeros(0, np.int64), np.zeros(0, np.int8))
    got = {int(a): int(b) for a, b in zip(lt, d)}
    s = 1 if side == "G" else -1
    for j in range(k + 1):
        t = slot + engine3.OFF + j * tf * 60
        if t not in got:
            if now < slot + (j + 1) * tf * 60 + GRACE:
                return None, None, True
            return "unknown", -1, False
        if got[t] == s or got[t] == 0:    # doji counts as a win (same as Engine 4)
            return "win", j, False
    return "loss", -1, False


def fmt_alert(r, slot, streak, hrs, labels):
    side = UP if r["side"] == "G" else DOWN
    mtg = "নেই" if r["mtg"] == 0 else "সর্বোচ্চ %d ধাপ (পরের %d ক্যান্ডেল)" % (r["mtg"], r["mtg"])
    done = ", ".join("%02d:%02d" % (h, r["minute"]) for h in hrs[-max(r["n"], 1):])
    return ("🎯 সিগন্যাল রাডার ১১\n"
            "💱 %s   ⏰ %s (UTC+6)\n"
            "%s\n"
            "⏱ টাইম ফ্রেম: %d মিনিট\n"
            "🔁 MTG: %s\n"
            "📊 ইতিহাস: %d/%d (১০০%%)\n"
            "ℹ️ আজ এই স্লটে পরপর %d লস: %s" % (
                pairs.pretty(r["pair"]), live.hhmm(slot), side, r["tf"], mtg, r["win"], r["sig"], streak, done))


async def alert_tick(send):
    if not _cfg["on"] or not _cfg["chat"] or not _state["rules"]:
        return
    now = time.time()
    DIAG["tick_at"] = now
    slot = int(now // 60 + 1) * 60
    loc = slot + engine3.OFF
    hour, minute = (loc % 86400) // 3600, (loc % 3600) // 60
    rules = [r for r in _state["rules"] if r["minute"] == minute and HOURS[0] <= hour <= HOURS[-1]
             and not pairs.is_otc(r["pair"])]        # OTC has no live candles (only the real-market list is live)
    if not rules:
        return                            # nothing to do -> no request to OANDA
    if live.market_closed(now):
        DIAG["skipped"] = "মার্কেট বন্ধ (সপ্তাহান্ত)"
        return
    cache = {}
    for r in rules:
        name = pairs.pretty(r["pair"])
        try:
            if r["pair"] not in cache:
                cache[r["pair"]] = await oanda.fetch_oc(r["pair"], live.day_start(slot), slot)
                DIAG["requests"] += 1
            ts, o, c = cache[r["pair"]]
            if len(ts) == 0 or ts[-1] < slot - STALE:
                DIAG["skipped"] = "%s: OANDA-র ক্যান্ডেল পুরনো/মার্কেট বন্ধ" % name
                continue
            info = streak_info(ts, o, c, r, hour)
            if info is None or not info[0]:
                continue
            key = "%s|%d|%s|%d|%d" % (r["pair"], r["minute"], r["side"], r["tf"], slot)
            if key in _alerts:
                continue
            _alerts[key] = dict(key=key, slot_ts=slot, pair=r["pair"], side=r["side"], mtg=r["mtg"], tf=r["tf"],
                                state="open", step=None)
            await send(_cfg["chat"], fmt_alert(r, slot, info[1], info[2], {}))
            DIAG["last_alert"] = (time.time(), name, live.hhmm(slot))
            try:
                await asyncio.to_thread(_save_alerts)
            except Exception:  # noqa
                log.exception("radar alert save failed (the alert itself was sent)")
        except Exception as e:  # noqa
            DIAG["skipped"] = "%s: %s" % (name, e)
            log.exception("radar alert for %s failed", name)


# ---------------------------------------------------------------- 3) results
async def result_tick(send):
    opens = [a for a in _alerts.values() if a["state"] == "open"]
    DIAG["res_at"] = time.time()
    now = time.time()
    for a in opens:
        end = a["slot_ts"] + (a["mtg"] + 1) * a["tf"] * 60
        if now < end + 3:
            continue
        try:
            ts, o, c = await oanda.fetch_oc(a["pair"], a["slot_ts"], end)
            DIAG["requests"] += 1
        except Exception:  # noqa
            log.exception("radar result read failed")
            continue
        state, step, wait = judge(ts, o, c, a["slot_ts"], a["mtg"], a["tf"], a["side"], now)
        if wait or state is None:
            continue
        a["state"], a["step"] = state, step
        try:
            await asyncio.to_thread(_save_alerts)
        except Exception:  # noqa
            log.exception("radar close save failed")
        if not _cfg["chat"]:
            continue
        w, l, u, _ = tally(live.day_start(a["slot_ts"]))
        head = "%s | %s | %s" % (pairs.pretty(a["pair"]), live.hhmm(a["slot_ts"]), "UP" if a["side"] == "G" else "DOWN")
        if state == "win":
            txt = "✅ WIN (%s)\n%s" % ("Non-MTG" if step == 0 else "MTG%d" % step, head)
        elif state == "loss":
            txt = "❌ LOSS\n%s" % head
        else:
            txt = "❔ রেজাল্ট জানা গেল না (ক্যান্ডেল আসেনি)\n%s" % head
        txt += "\n📅 আজকের রাডার ১১ হিসাব: ✅ %d • ❌ %d%s" % (w, l, (" • ❔ %d" % u) if u else "")
        await send(_cfg["chat"], txt)


# ---------------------------------------------------------------- loops
async def _sleep_to(sec_of_minute):
    now = time.time()
    t = int(now // 60) * 60 + sec_of_minute
    if t <= now + 0.5:
        t += 60
    await asyncio.sleep(t - now)


async def alert_loop(send):
    while True:
        await _sleep_to((60 - LEAD) % 60)
        try:
            await alert_tick(send)
        except Exception:  # noqa
            log.exception("radar alert tick")


async def result_loop(send):
    while True:
        await _sleep_to(RES_SEC)
        try:
            await result_tick(send)
        except Exception:  # noqa
            log.exception("radar result tick")


def _bdt_date(t):
    return dt.datetime.fromtimestamp(t, pairs.BDT).date()


async def refresh_loop(refresh_cb):
    """Once a day (from REFRESH_HM, BDT) rebuilds the radar list with the saved settings - only while the radar is ON."""
    while True:
        await asyncio.sleep(30)
        try:
            now = time.time()
            n = dt.datetime.fromtimestamp(now, pairs.BDT)
            if (not _cfg["on"] or not _state["meta"].get("pairs") or live.market_closed(now)
                    or (n.hour, n.minute) < REFRESH_HM or n.hour >= 23
                    or _bdt_date(_state["built_at"] or 0) == n.date() or now - DIAG["try_at"] < 600):
                continue
            DIAG["try_at"] = now
            await refresh_cb()
        except Exception as e:  # noqa
            DIAG["refresh_err"] = str(e)
            log.exception("radar refresh")

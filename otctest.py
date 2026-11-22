"""/otctest (OTC) and /realtest (real market) — replay a saved /engine10 radar list on the stored candles of chosen day(s), like a live run (no live data needed).

The list is the TEXT of an /engine10 result that the user pastes into the chat, so the list is frozen (it can not be changed by
the candles of the test day). The bot reads tf, MTG, N back-to-back losses ("exactly N" / "N or more") and every pair's
green / red minutes from that text, asks for the day(s), reads the M1 candles of that day from Neon (read-only, via source.load_oc),
and replays every slot with the SAME code as the back-test (engine4.outcomes): each hour the slot's minute is traded, N losses in a
row (the streak restarts every day) -> the next hour's trade is a SIGNAL; its result is a direct win / MTG win / loss (doji = win).
Messages: signal, then result (in order of time), then a summary + CSV. Optional separate chats: env OTC_TEST_CHAT_ID (OTC pairs) and
REAL_TEST_CHAT_ID (real-market pairs); a market without its own chat sends to the chat where /otctest was started.

Nothing of the other engines is touched; this file only registers its own command, callbacks and a text handler.
"""
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


import asyncio
import csv
import datetime as dt
import html
import io
import logging
import os
import re
import time

import numpy as np
from telegram import BotCommand, InlineKeyboardButton as Btn, InlineKeyboardMarkup as Markup  # noqa: F401
from telegram.error import BadRequest
from telegram.ext import ApplicationHandlerStop, CallbackQueryHandler, CommandHandler, MessageHandler, filters

e3 = engine3
pass  # engine4 is embedded below
import pairs
import source

log = logging.getLogger("otctest")
BDT = pairs.BDT
OFF = e3.OFF
TEST_CHAT = {"otc": os.environ.get("OTC_TEST_CHAT_ID", "").strip(),      # optional channel / group for the OTC replay messages
             "real": os.environ.get("REAL_TEST_CHAT_ID", "").strip()}   # optional channel / group for the real-market replay messages
UP, DOWN = "UP (Call) 📈", "DOWN (Put) 📉"
MAX_DAYS = 7                                                          # most days in one test

def _new():
    return dict(step=None, text="", rules=None, hdr=None, mode="all", stop=False, running=False, dl=None)     # step: None | "paste" | "date"


S = {"otc": _new(), "real": _new()}     # everything is kept apart per market: /otctest -> S["otc"], /realtest -> S["real"]
LAST = {"m": None}                      # market whose text input is awaited most recently
D = {}          # injected from bot.py: send, allowed, deny, lock
NAME = {"otc": "🧪 OTC", "real": "📈 রিয়েল"}
CMD = {"otc": "/otctest", "real": "/realtest"}
PFX = {"otc": "ot", "real": "rt"}


def _reset(m):
    S[m].update(step=None, text="", rules=None, hdr=None, stop=False, dl=None)


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


# ---------------------------------------------------------------- 1b) read a DAY LIST (result of /engine12, /engine14 or /otctest itself)
# Second accepted format (the /engine10 text above still works exactly as before):
#   📅 - 2026.11.21
#   M1 AUDNZD-OTC 02:05 PUT ❌        (✅ = direct win, ✅¹ = MTG1 win, ✅² = MTG2 win, ❌ = loss)
# Every distinct (pair, side, minute) of the rows becomes a radar slot (BUY = green, PUT = red). The list has no header, so MTG and the number
# of back-to-back losses are chosen with buttons after the list is read (the MTG default is the highest level seen in the rows).
_DL_ROW = re.compile(r"^\s*M(\d{1,2})\s+(\S+)\s+(\d{1,2}):(\d{2})(?!\d)\s*(BUY|PUT|UP|DOWN|CALL)?(.*)$", re.I)
_DL_DATE = re.compile(r"(?<!\d)(\d{4})\s*[-./]\s*(\d{1,2})\s*[-./]\s*(\d{1,2})(?!\d)")
_SUPLVL = {"\u00b9": 1, "\u00b2": 2, "\u00b3": 3}


def parse_daylist(text):
    """-> (dict(tf, rules=[(pair, side, minute)], lvl, dates, skipped) or None, error text)."""
    text = _digits(text).replace("\ufe0f", "")
    rows, dates, lvl = [], [], 0
    for ln in text.splitlines():
        d = _DL_DATE.search(ln)
        if d and not _DL_ROW.match(ln):
            try:
                dd = dt.date(int(d.group(1)), int(d.group(2)), int(d.group(3)))
                if dd not in dates:
                    dates.append(dd)
            except ValueError:
                pass
            continue
        m = _DL_ROW.match(ln)
        if not m:
            continue
        side = (m.group(5) or "").upper()
        side = {"BUY": "G", "UP": "G", "CALL": "G", "PUT": "R", "DOWN": "R"}.get(side)
        pair = pairs.norm(m.group(2))
        if not side or not pair:
            continue
        tf, mi = int(m.group(1)), int(m.group(4))
        rows.append((tf, pair, side, mi))
        w = re.search(r"\u2705\s*([\u00b9\u00b2\u00b3])", m.group(6) or "")
        if w:
            lvl = max(lvl, _SUPLVL[w.group(1)])
    if not rows:
        return None, "সিগন্যাল লিস্টের লাইন পাইনি (যেমন: M1 AUDNZD-OTC 02:05 PUT ✅)।"
    tfs = {}
    for r in rows:
        tfs[r[0]] = tfs.get(r[0], 0) + 1
    tf = max(tfs, key=lambda k: tfs[k])
    if tf <= 0 or 60 % tf:
        return None, "টাইম ফ্রেম বুঝিনি: M%d" % tf
    rules, seen, skipped = [], set(), 0
    for t_, pair, side, mi in rows:
        if t_ != tf or mi % tf:
            skipped += 1
            continue
        k = (pair, side, mi)
        if k not in seen:
            seen.add(k)
            rules.append(k)
    if not rules:
        return None, "কোনো স্লট পাইনি।"
    return dict(tf=tf, rules=rules, lvl=lvl, dates=sorted(dates), skipped=skipped), None


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


def _mk(pair):
    return "otc" if pairs.is_otc(pair) else "real"


def _dest(chat_id, market):
    """Where the replay messages of a market go: its own channel (env) or the chat that started the test."""
    v = TEST_CHAT.get(market) or chat_id
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


def split_market(sigs, info, rules, market):
    """Only the part of a replay that belongs to one market."""
    ss = [x for x in sigs if _mk(x["pair"]) == market]
    inf = dict(no_data=[x for x in info["no_data"] if _mk(x[0]) == market], partial=[x for x in info["partial"] if _mk(x[0]) == market],
               missing=info["missing_m"][market], missing_m=info["missing_m"])
    return ss, inf, [r for r in rules if _mk(r["pair"]) == market]


# ---------------------------------------------------------------- 3) replay (same code as the back-test)
def replay(rules, hdr, dates):
    """-> (signals list, info dict). signal = dict(ts, date, pair, side, minute, hour, mtg, res 1|-1, lvl, doji, streak, loss_hours).
    info: no_data[(pair, date)], partial[(pair, date, last 'HH:MM')], skipped_missing (int)."""
    tf, n, atleast = hdr["tf"], hdr["n"], hdr["atleast"]
    info = dict(no_data=[], partial=[], missing=0, missing_m={"otc": 0, "real": 0})
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
                            info["missing_m"][_mk(pair)] += 1
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
    return ("%s টেস্ট (রিপ্লে) • %s\n💱 %s   ⏰ %02d:%02d (UTC+6)\n%s\n⏱ টাইম ফ্রেম: %d মিনিট\n🔁 MTG: %s\nℹ️ এই স্লটে পরপর %d লস: %s" % (
        "🧪 OTC" if pairs.is_otc(s["pair"]) else "📈 রিয়েল", s["date"].strftime("%d %b"), pairs.pretty(s["pair"]), s["hour"], s["minute"], side, hdr["tf"], _mtg_txt(s["mtg"]), s["streak"], done))


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


SUP = {1: "¹", 2: "²", 3: "³"}


def _row_icon(s):
    if s["res"] != 1:
        return "❌"
    return "✅" + SUP.get(s["lvl"], "") if s["lvl"] else "✅"          # direct win = ✅ only, MTG1 win = ✅¹, MTG2 = ✅², MTG3 = ✅³


def _day_lists(sigs, hdr):
    """Per date: a header (📅 - 2026.11.16 + divider) and the day's signals in time order, one per row:
    'M5 USCrude-OTC 03:45 BUY ✅'. Rows are put into <pre> blocks (monospace) of <= 60 rows so a message never cuts a block in half."""
    width = max(len(pairs.pretty(s["pair"])) for s in sigs)
    out = []
    for d in sorted({s["date"] for s in sigs}):
        rows = ["M%d %s %02d:%02d %s %s" % (hdr["tf"], pairs.pretty(s["pair"]).ljust(width), s["hour"], s["minute"],
                                           ("BUY" if s["side"] == "G" else "PUT"), _row_icon(s))
                for s in sorted((x for x in sigs if x["date"] == d), key=lambda x: (x["hour"], x["minute"], x["pair"]))]
        out.append("📅 - %s\n▬▬▬▬▬▬▬▬▬ • ▬▬▬▬▬▬▬▬▬" % d.strftime("%Y.%m.%d"))
        for i in range(0, len(rows), 60):
            out.append(("PRE", "<pre>%s</pre>" % html.escape("\n".join(rows[i:i + 60]), quote=False)))
    return out


def summary_texts(sigs, info, hdr, dates, rules, partial_note=True):
    n, w, l, lv = stats(sigs)
    ds = ", ".join(d.strftime("%d %b") for d in dates)
    L = ["📊 %s টেস্ট রিপোর্ট • %s" % ("OTC" if rules and pairs.is_otc(rules[0]["pair"]) else "রিয়েল মার্কেট", ds), summary_line(hdr, rules), ""]
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
        # per-day signal list (replaces the old "পেয়ার ধরে" block): date, then  M<tf> PAIR HH:MM ✅ / ✅¹ (MTG1 win) / ❌
        L.append("")
        L += _day_lists(sigs, hdr)
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
    # the messages are sent with parse_mode="HTML" (the signal rows sit in <pre> = monospace, so the columns line up)
    ents = [x[1] if isinstance(x, tuple) else html.escape(x, quote=False) for x in L]
    out, cur = [], ""
    for ent in ents:
        if cur and len(cur) + len(ent) + 1 > 3900:
            out.append(cur)
            cur = ""
        cur += ent + "\n"
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


# ---------------------------------------------------------------- 5) Telegram glue (one set per market)
def _kb_paste(m):
    p = PFX[m]
    return Markup([[Btn("✅ পাঠানো শেষ", callback_data="%s:done:0" % p), Btn("❌ বাতিল", callback_data="%s:cancel:0" % p)]])


def _kb_date(m):
    p = PFX[m]
    return Markup([[Btn("📨 মেসেজ: সব (লাইভের মতো)" if S[m]["mode"] == "all" else "📊 মেসেজ: শুধু সারাংশ", callback_data="%s:mode:0" % p)],
                   [Btn("গতকাল", callback_data="%s:day:y" % p), Btn("আজ", callback_data="%s:day:t" % p)],
                   [Btn("❌ বাতিল", callback_data="%s:cancel:0" % p)]])


DL_MTG = {0: "নন-MTG", 1: "MTG1", 2: "MTG2", 3: "MTG3"}


def _dt(on, label):
    return ("✅ " if on else "") + label


def _kb_dl(m):
    p, d = PFX[m], S[m]["dl"]
    return Markup([[Btn(_dt(d["mtg"] == v, lb), callback_data="%s:dm:%d" % (p, v)) for v, lb in DL_MTG.items()],
                   [Btn(_dt(d["n"] == v, "লস %d" % v), callback_data="%s:dn:%d" % (p, v)) for v in range(1, 7)]
                   + [Btn(_dt(d["n"] > 6, "✏️ %d" % d["n"] if d["n"] > 6 else "✏️"), callback_data="%s:dn:custom" % p)],
                   [Btn(_dt(not d["atleast"], "ঠিক N লস"), callback_data="%s:dx:0" % p), Btn(_dt(d["atleast"], "N বা বেশি"), callback_data="%s:dx:1" % p)],
                   [Btn("▶️ চালিয়ে যাও", callback_data="%s:go:0" % p), Btn("❌ বাতিল", callback_data="%s:cancel:0" % p)]])


def _dl_text(m):
    d = S[m]["dl"]
    g = sum(1 for r in d["rules"] if r[1] == "G")
    ds = (" • তারিখ: " + ", ".join(x.strftime("%d %b") for x in d["dates"])) if d["dates"] else ""
    return ("%s: দিনের সিগন্যাল লিস্ট পড়লাম (ইঞ্জিন ১০-এর হেডার নেই)\n"
            "• টাইম ফ্রেম %d মিনিট • %d টা স্লট (🟢 %d, 🔴 %d) • %d পেয়ার%s\n\n"
            "লিস্টে যেটা লেখা থাকে না সেটা বেছে নাও, তারপর ▶️ চাপো:\n"
            "• MTG: %s\n• পরপর লস: %d টা (%s)") % (
        NAME[m], d["tf"], len(d["rules"]), g, len(d["rules"]) - g, len({r[0] for r in d["rules"]}), ds,
        DL_MTG[d["mtg"]], d["n"], "N বা বেশি" if d["atleast"] else "ঠিক N")


async def _cmd(update, m):
    if not D["allowed"](update):
        return await D["deny"](update)
    st = S[m]
    if st["running"]:
        return await update.effective_message.reply_text("%s টেস্ট চলছে। শেষ হওয়া পর্যন্ত অপেক্ষা করো (বা মেসেজের ⏹ বাটন চাপো)।" % NAME[m])
    if m == "otc" and not source.otc_enabled():
        return await update.effective_message.reply_text("OTC_DATABASE_URL_* সেট করা নেই।")
    if m == "real" and not source.enabled():
        return await update.effective_message.reply_text("DATABASE_URL_* সেট করা নেই।")
    _reset(m)
    st["step"] = "paste"
    LAST["m"] = m
    await update.effective_message.reply_text(
        "%s টেস্ট (রিপ্লে)\n\n/engine10-এর %s তালিকার রেজাল্ট টেক্সট এখানে পাঠাও (কপি করে পেস্ট)। একাধিক মেসেজ হলে সবগুলো পাঠাও, তারপর ✅ চাপো।\n"
        "প্রথম মেসেজে হেডার লাইন (টাইম ফ্রেম, MTG, পরপর লস) থাকতে হবে।\n\n"
        "অথবা ইঞ্জিন ১২/১৪-এর দিনের সিগন্যাল লিস্ট (📅 তারিখ + M1 PAIR 02:05 PUT ✅ ...) পাঠাতে পারো; তখন MTG আর লস সংখ্যা বাটনে বেছে নেবে।" % (NAME[m], "OTC" if m == "otc" else "রিয়েল মার্কেটের"), reply_markup=_kb_paste(m))


async def cmd_otctest(update, ctx):
    await _cmd(update, "otc")


async def cmd_realtest(update, ctx):
    await _cmd(update, "real")


async def on_text(update, ctx):
    """Group -1 handler: consumes text only while /otctest or /realtest waits for it, otherwise lets the normal handlers run."""
    waiting = [m for m in ("otc", "real") if S[m]["step"] is not None]
    if not waiting or not D["allowed"](update):
        return
    m = LAST["m"] if LAST["m"] in waiting else waiting[0]
    st = S[m]
    msg = update.effective_message
    txt = msg.text or ""
    if st["step"] == "paste":
        st["text"] += txt + "\n"
        await msg.reply_text("%s: পেয়েছি (এ পর্যন্ত %d লাইন)। আরও থাকলে পাঠাও, শেষ হলে ✅ চাপো।" % (NAME[m], st["text"].count("\n")), reply_markup=_kb_paste(m))
        raise ApplicationHandlerStop
    if st["step"] == "dln":
        mm = re.search(r"\d+", _digits(txt))
        if not mm or not (1 <= int(mm.group(0)) <= 20):
            await msg.reply_text("১ থেকে ২০-এর মধ্যে একটা সংখ্যা লেখো (যেমন: 7)।")
            raise ApplicationHandlerStop
        st["dl"]["n"] = int(mm.group(0))
        st["step"] = "dl"
        await msg.reply_text(_dl_text(m), reply_markup=_kb_dl(m))
        raise ApplicationHandlerStop
    if st["step"] == "date":
        today = dt.datetime.now(BDT).date()
        ds = parse_dates(txt, today)
        if not ds:
            await msg.reply_text("তারিখ বুঝিনি। এভাবে লেখো: 17 / 17,18,19 / 17-19 / 17/11 / 2026-11-17 / গতকাল", reply_markup=_kb_date(m))
            raise ApplicationHandlerStop
        await _start(ctx, msg.chat_id, ds, m)
        raise ApplicationHandlerStop


async def _start(ctx, chat_id, dates, m):
    st = S[m]
    today = dt.datetime.now(BDT).date()
    if any(d > today for d in dates):
        return await D["send"](chat_id, "ভবিষ্যতের তারিখ দিয়েছ। আজ বা আগের তারিখ দাও।")
    if len(dates) > MAX_DAYS:
        return await D["send"](chat_id, "একসাথে সর্বোচ্চ %d দিন। কম দিন দাও।" % MAX_DAYS)
    if st["running"]:
        return
    st["step"] = None
    st["running"] = True
    st["stop"] = False
    ctx.application.bot_data["otctest_job_" + m] = asyncio.create_task(_job(chat_id, list(dates), m))


async def _job(chat_id, dates, m):
    send = D["send"]
    st = S[m]
    hdr, rules = st["hdr"], st["rules"]
    dest = _dest(chat_id, m)
    delay = 3.2 if str(dest) != str(chat_id) else 1.1
    kb_stop = Markup([[Btn("⏹ বন্ধ করো", callback_data="%s:stop:0" % PFX[m])]])
    try:
        prog = await send(chat_id, "⏳ %s টেস্ট: %s-এর ক্যান্ডেল পড়ছি ও রিপ্লে করছি…" % (NAME[m], ", ".join(d.strftime("%d %b") for d in dates)), reply_markup=kb_stop)
        async with D["lock"]:
            sigs, info = await asyncio.to_thread(replay, rules, hdr, dates)
        if st["mode"] == "all" and sigs:
            if str(dest) != str(chat_id):
                await send(chat_id, "📨 %d টা সিগন্যালের মেসেজ চ্যানেলে যাচ্ছে (প্রায় %d মিনিট লাগবে)।" % (len(sigs), max(1, round(len(sigs) * 2 * delay / 60))))
            tally = {}
            last_day = None
            for s in sigs:
                if st["stop"]:
                    await send(chat_id, "⏹ বন্ধ করা হয়েছে।")
                    break
                if s["date"] != last_day:
                    last_day = s["date"]
                    await send(dest, "━━━━━━━━━━━━\n📅 %s-এর রিপ্লে শুরু (%s)" % (s["date"].strftime("%d %b %Y"), NAME[m]))
                    await asyncio.sleep(delay)
                await send(dest, fmt_signal(s, hdr))
                await asyncio.sleep(delay)
                t = tally.setdefault(s["date"], [0, 0])
                t[0 if s["res"] == 1 else 1] += 1
                await send(dest, fmt_result(s, t))
                await asyncio.sleep(delay)
        for t in summary_texts(sigs, info, hdr, dates, rules):
            await send(dest, t, parse_mode="HTML")
            if str(dest) != str(chat_id):
                await send(chat_id, t, parse_mode="HTML")
        if sigs:
            bio = io.BytesIO(make_csv(sigs))
            bio.name = "%stest_%s.csv" % (PFX[m] if m == "otc" else "real", dt.datetime.now(BDT).strftime("%Y%m%d_%H%M"))
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
        await send(chat_id, "❌ %s টেস্ট হয়নি: %s" % (NAME[m], e))
    finally:
        st["running"] = False


async def on_cb(update, ctx):
    q = update.callback_query
    if not D["allowed"](update):
        return await q.answer("অনুমতি নেই", show_alert=True)
    pfx, key, val = q.data.split(":")[:3]
    m = "real" if pfx == "rt" else "otc"
    st = S[m]
    if key == "stop":
        st["stop"] = True
        return await q.answer("বন্ধ করছি…")
    if key == "cancel":
        _reset(m)
        await q.answer("বাতিল")
        try:
            await q.edit_message_text("%s টেস্ট বাতিল।" % NAME[m])
        except BadRequest:
            pass
        return
    if key == "done":
        if st["step"] != "paste":
            return await q.answer("আগে %s দাও" % CMD[m], show_alert=True)
        hdr, rules, err = parse(st["text"])
        dl = None
        if err:                                          # no /engine10 header -> maybe a day list of /engine12, /engine14 or /otctest
            dl, _derr = parse_daylist(st["text"])
            if dl:
                err = None
                rules = [dict(pair=p_, side=s_, minute=mi_, mtg=1) for p_, s_, mi_ in dl["rules"]]
        if not err:
            other = [r for r in rules if _mk(r["pair"]) != m]
            if other:
                if len(other) == len(rules):
                    err = "এটা %s তালিকা। %s দাও।" % ("রিয়েল" if m == "otc" else "OTC", CMD["real" if m == "otc" else "otc"])
                else:
                    err = "তালিকায় রিয়েল ও OTC মিশে আছে। আলাদা আলাদা তালিকা দিয়ে %s আর %s চালাও।" % (CMD["otc"], CMD["real"])
        if err:
            return await q.answer(err[:190], show_alert=True)
        if dl:
            dl.update(mtg=max(dl["lvl"], 1), n=3, atleast=False)
            st["dl"], st["step"] = dl, "dl"
            LAST["m"] = m
            await q.answer()
            await q.message.reply_text(_dl_text(m), reply_markup=_kb_dl(m))
            return
        st["hdr"], st["rules"], st["step"] = hdr, rules, "date"
        LAST["m"] = m
        await q.answer()
        await q.message.reply_text("📋 %s তালিকা পড়লাম:\n%s\n\nকোন তারিখে টেস্ট করবে? লেখো: 17 / 17,18,19 / 17-19 / 17/11 / গতকাল\n(তালিকা বানানোর পরের দিনের তারিখ দাও, নইলে ফলাফল চিট হবে)" % (
            NAME[m], summary_line(hdr, rules)), reply_markup=_kb_date(m))
        return
    if key in ("dm", "dn", "dx", "go"):
        if st["step"] not in ("dl", "dln") or not st["dl"]:
            return await q.answer("আগে %s দিয়ে লিস্ট পাঠাও" % CMD[m], show_alert=True)
        d = st["dl"]
        if key == "dm":
            d["mtg"] = int(val)
        elif key == "dx":
            d["atleast"] = (val == "1")
        elif key == "dn":
            if val == "custom":
                st["step"] = "dln"
                await q.answer()
                return await q.message.reply_text("✏️ পরপর কয়টা লস হলে সিগন্যাল ধরবে লেখো (১ থেকে ২০, যেমন: 7)।")
            d["n"] = int(val)
        elif key == "go":
            hdr = dict(tf=d["tf"], n=d["n"], atleast=d["atleast"], mtgs=[d["mtg"]])
            rules = [dict(pair=p_, side=s_, minute=mi_, mtg=d["mtg"]) for p_, s_, mi_ in d["rules"]]
            st["hdr"], st["rules"], st["step"] = hdr, rules, "date"
            await q.answer()
            return await q.message.reply_text("📋 %s তালিকা পড়লাম:\n%s\n\nকোন তারিখে টেস্ট করবে? লেখো: 17 / 17,18,19 / 17-19 / 17/11 / গতকাল\n(তালিকা বানানোর পরের দিনের তারিখ দাও, নইলে ফলাফল চিট হবে)" % (
                NAME[m], summary_line(hdr, rules)), reply_markup=_kb_date(m))
        await q.answer()
        try:
            await q.edit_message_text(_dl_text(m), reply_markup=_kb_dl(m))
        except BadRequest:
            pass
        return
    if key == "mode":
        st["mode"] = "sum" if st["mode"] == "all" else "all"
        await q.answer()
        try:
            await q.edit_message_reply_markup(reply_markup=_kb_date(m))
        except BadRequest:
            pass
        return
    if key == "day":
        if st["step"] != "date":
            return await q.answer("আগে %s দিয়ে তালিকা পাঠাও" % CMD[m], show_alert=True)
        today = dt.datetime.now(BDT).date()
        await q.answer("শুরু হচ্ছে…")
        await _start(ctx, q.message.chat_id, [today - dt.timedelta(days=1)] if val == "y" else [today], m)


def register(app, send, allowed, deny, lock, bot_getter):
    """Called once from bot.py main(): own commands (/otctest, /realtest), callbacks and a text handler (group -1) — nothing existing is changed."""
    D.update(send=send, allowed=allowed, deny=deny, lock=lock, bot=bot_getter)
    app.add_handler(CommandHandler("otctest", cmd_otctest))
    app.add_handler(CommandHandler("realtest", cmd_realtest))
    app.add_handler(CallbackQueryHandler(on_cb, pattern=r"^(ot|rt):"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text), group=-1)


COMMAND = BotCommand("otctest", "OTC টেস্ট: /engine10 OTC তালিকা পেস্ট করে কোনো দিনের রিপ্লে")
COMMAND_REAL = BotCommand("realtest", "রিয়েল টেস্ট: /engine10 রিয়েল তালিকা পেস্ট করে কোনো দিনের রিপ্লে")

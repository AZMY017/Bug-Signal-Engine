"""Engine 9 — Engine 8 + Engine 4 in one step (no database, no Telegram code here).

1) Engine 8 part: for every pair, find the minutes that share the LOWEST one-day back-to-back loss at the chosen MTG,
   separately for GREEN and for RED (same data / candle / MTG / doji rules as Engine 3 / 6 / 8).
2) Engine 4 part: take exactly those minutes (GREEN minutes -> GREEN trades, RED minutes -> RED trades) and run the
   "signal after N back-to-back losses" counting of Engine 4 on them (doji = win, same as Engine 4).
Output is a short Telegram text: per side and per minute -> signals, win, loss, win rate, doji wins, plus a total line.
"""
import engine3 as e3
import engine4
import engine6
import engine8

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
                out, lvl, dj = engine4.outcomes(res["dirm"], res["val"], tf, m, res["days"], side, k)
                sigs, miss = engine4.find_signals(out, lvl, dj, n_loss, scope, atleast)
                sw = sum(1 for x in sigs if x[2] == 1)
                rows.append(dict(minute=m, sig=len(sigs), win=sw, loss=len(sigs) - sw,
                                 doji=sum(1 for x in sigs if x[2] == 1 and x[4]), missing=miss))
            sides[side] = dict(lo=lo, rows=rows)
        per[k] = sides
    return dict(pair=pair, tf=tf, months=months, d0=res["d0"], days=res["days"], ks=ks, n_loss=n_loss,
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

"""Engine 7 — Engine 6's numbers, but shown as Telegram buttons instead of a PDF (no database, no Telegram code here).

Engine 7 reuses engine6 / engine3 1:1 (same data, same candle / MTG / doji rules, same B2B day / chain numbers, same ranking).
Only the output differs: for every pair / MTG step / side (GREEN or RED) we return the ranked rows, and the bot shows the
minutes that share the LOWEST one-day back-to-back loss as ticked buttons.
"""
import engine6

MAXK = engine6.MAXK
MODES = engine6.MODES
SIDES = ("G", "R")


def analyze(pair, ts, o, c, tf, months, minutes):
    """Same as Engine 6 / Engine 3."""
    return engine6.analyze(pair, ts, o, c, tf, months, minutes)


def collect(res, levels):
    """-> {k: {'G': ranked rows, 'R': ranked rows}} for every chosen MTG level (rows are already sorted, rank 1 first)."""
    return {k: {s: engine6.ranked(res, k, s) for s in SIDES} for k in sorted(set(levels))}


def lowest(rows):
    """-> (lowest one-day B2B loss, [minutes that have it, in rank order]); (None, []) when there are no rows."""
    if not rows:
        return None, []
    lo = rows[0]["b2b_day"]          # rows are sorted by b2b_day first
    return lo, [r["minute"] for r in rows if r["b2b_day"] == lo]


def popup(pair_txt, k, side, r):
    """Short text (<200 chars) for the alert that opens when a minute button is pressed."""
    bd = " + ".join(str(v) for v in r["levels"][:k + 1])
    return ("%s • %s • %s :%02d\nর‍্যাংক %d\nউইন %d • লস %d • WR %.1f%%\nB2B দিন %d • চেইন %d\nউইন ভাঙা: %s" % (
        pair_txt, MODES[k], "গ্রিন" if side == "G" else "রেড", r["minute"], r["rank"], r["win"], r["loss"],
        100 * r["wr"], r["b2b_day"], r["b2b_chain"], bd))[:195]

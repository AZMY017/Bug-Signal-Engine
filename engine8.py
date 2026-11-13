"""Engine 8 — Engine 6's numbers, but sent as plain Telegram TEXT (no PDF, no buttons, no database, no Telegram code here).

Engine 8 reuses engine6 / engine3 1:1 (same data, same candle / MTG / doji rules, same B2B day / chain numbers, same ranking).
Only the output differs: for every pair / MTG step we return one short text with two headings, GREEN and RED, and under each
heading the minutes that share the LOWEST one-day back-to-back loss.
"""
import engine6
import engine7

MAXK = engine6.MAXK
MODES = engine6.MODES
LIMIT = 4000        # Telegram's hard limit is 4096 characters per message


def analyze(pair, ts, o, c, tf, months, minutes):
    """Same as Engine 6 / Engine 3."""
    return engine6.analyze(pair, ts, o, c, tf, months, minutes)


def collect(res, levels):
    """-> {k: {'G': ranked rows, 'R': ranked rows}} (same as Engine 7)."""
    return engine7.collect(res, levels)


def minutes_line(rows):
    """-> (lowest one-day B2B loss, ':05 :12 :30') for the minutes that share it, sorted by time; (None, '—') when no data."""
    lo, mins = engine7.lowest(rows)
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

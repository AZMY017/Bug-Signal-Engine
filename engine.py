"""Bug-signal scan: pure numpy, no database, no Telegram.

A "bug" = a fixed time slot (pair, minute-of-hour, direction) inside an hour window (UTC+6) whose candle goes the same
way very often. One signal per hour of the window, every day. Win = the slot candle goes the chosen way; with MTG k
the next k one-minute candles get a second/third chance. Doji (open == close) is not a win.

Back-to-back losses are counted inside one day (the streak restarts every day) over the signals of that window.
For every slot we also measure the "after N losses" rule: take the next signal only when exactly N signals just lost in
a row (N = 0 means: take every signal). The days are cut into a train part (first 2/3) and a test part (last 1/3);
a pattern is called robust only when the test part confirms it.
"""
import math

import numpy as np

OFF = 6 * 3600       # UTC+6
NMAX = 4             # "after N losses": N = 0..4
MTGS = (0, 1, 2)
WINDOWS_DEFAULT = [(0, 24), (0, 12), (12, 24), (0, 6), (6, 12), (12, 18), (18, 24)]


def base_rate(k):
    """Win rate a coin flip would get with k MTG steps (the pattern must beat this)."""
    return 1.0 - 0.5 ** (k + 1)


def wilson_lb(w, n, z=1.96):
    if n <= 0:
        return 0.0
    p = w / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - r) / den)


def build_matrices(ts, d):
    """ts (unix seconds), d (+1/-1/0) -> (first_day, dirm (D,1443) int8, val (D,1443) bool); day/minute in UTC+6."""
    loc = ts + OFF
    day = loc // 86400
    mi = (loc % 86400) // 60
    d0 = int(day.min())
    nd = int(day.max()) - d0 + 1
    dirm = np.zeros((nd, 1443), np.int8)
    val = np.zeros((nd, 1443), bool)
    r = (day - d0).astype(np.int64)
    dirm[r, mi] = d
    val[r, mi] = True
    return d0, dirm, val


def outcome_cubes(dirm, val, side, k):
    """-> W (D,24,60) win flags, V (D,24,60) usable flags, for signal at hour h minute m."""
    eq = dirm == side
    w = eq[:, 0:1440].copy()
    v = val[:, 0:1440].copy()
    for j in range(1, k + 1):
        w |= eq[:, j:1440 + j]
        v &= val[:, j:1440 + j]
    nd = w.shape[0]
    return (w & v).reshape(nd, 24, 60), v.reshape(nd, 24, 60)


def streak_pass(w, v, hs, he):
    """w, v: (D,24,60). Returns n[N,60], wins[N,60], maxrun[60] for hours hs..he-1."""
    wh = w[:, hs:he, :]
    vh = v[:, hs:he, :]
    nd, hh, mm = wh.shape
    run = np.zeros((nd, mm), np.int64)
    n = np.zeros((NMAX + 1, mm), np.int64)
    wins = np.zeros((NMAX + 1, mm), np.int64)
    maxrun = np.zeros(mm, np.int64)
    for h in range(hh):
        vv = vh[:, h, :]
        ww = wh[:, h, :]
        n[0] += vv.sum(0)
        wins[0] += ww.sum(0)
        for k in range(1, NMAX + 1):
            mk = vv & (run == k)
            n[k] += mk.sum(0)
            wins[k] += (mk & ww).sum(0)
        run = np.where(vv, np.where(ww, 0, run + 1), run)
        np.maximum(maxrun, run.max(0), out=maxrun)
    return n, wins, maxrun


def scan_pair(pair, ts, d, windows, p):
    """p: dict(min_wr, min_edge, z, min_n, min_n_cond, min_test_n, min_test_n_cond, split).
    Returns the list of row dicts that pass. If p["near_out"] is a list, the 5 closest misses of this pair are
    appended to it too (so you can see what the scan found even when nothing passes)."""
    if len(ts) < 5000:
        return []
    _, dirm, val = build_matrices(ts, d)
    nd = dirm.shape[0]
    cut = max(1, int(nd * p["split"]))
    rows = []
    near = []
    near_out = p.get("near_out")
    z = p.get("z", 3.5)
    for side, sname in ((1, "UP"), (-1, "DOWN")):
        for k in MTGS:
            W, V = outcome_cubes(dirm, val, side, k)
            base = base_rate(k)
            for hs, he in windows:
                ntr, wtr, mtr = streak_pass(W[:cut], V[:cut], hs, he)
                nte, wte, mte = streak_pass(W[cut:], V[cut:], hs, he)
                n = ntr + nte
                w = wtr + wte
                mall = np.maximum(mtr, mte)
                for N in range(NMAX + 1):
                    need_n = p["min_n"] if N == 0 else p["min_n_cond"]
                    cand = np.nonzero(n[N] >= need_n)[0]
                    for m in cand:
                        nn, ww = int(n[N, m]), int(w[N, m])
                        wr = ww / nn
                        lb = wilson_lb(ww, nn, z)
                        score = lb - base
                        passes = wr >= p["min_wr"] and score >= p["min_edge"]
                        track = near_out is not None and (len(near) < 5 or score > near[-1]["score"])
                        if not passes and not track:
                            continue
                        a, b = int(nte[N, m]), int(wte[N, m])
                        need_t = p["min_test_n"] if N == 0 else p["min_test_n_cond"]
                        robust = bool(passes and a >= need_t and b / a >= p["min_wr"] - 0.05 and b / a >= base + 0.02
                                      and int(wtr[N, m]) / max(1, int(ntr[N, m])) >= base + 0.02)
                        row = dict(pair=pair, minute=int(m), side=sname, mtg=k, hs=hs, he=he, nloss=N,
                                   n=nn, w=ww, ntr=int(ntr[N, m]), wtr=int(wtr[N, m]), nte=a, wte=b,
                                   maxl=int(mall[m]), maxl_tr=int(mtr[m]), maxl_te=int(mte[m]),
                                   base=base, score=score, robust=robust)
                        if passes:
                            rows.append(row)
                        elif track:
                            near.append(row)
                            near.sort(key=lambda r: r["score"], reverse=True)
                            del near[5:]
    if near_out is not None:
        near_out.extend(near)
    return rows


def check_slot(ts, d, side, minute, hs, he):
    """Plain numbers for one slot (for /check): for MTG 0/1/2 -> list of dicts with n, wins, max back-to-back loss,
    and the 'after N losses' win rates, over all days and over the train/test halves."""
    _, dirm, val = build_matrices(ts, d)
    nd = dirm.shape[0]
    cut = max(1, int(nd * 2 / 3))
    out = []
    for k in MTGS:
        W, V = outcome_cubes(dirm, val, side, k)
        ntr, wtr, mtr = streak_pass(W[:cut], V[:cut], hs, he)
        nte, wte, mte = streak_pass(W[cut:], V[cut:], hs, he)
        n, w = ntr + nte, wtr + wte
        out.append(dict(mtg=k, base=base_rate(k), maxl=int(max(mtr[minute], mte[minute])),
                        n=[int(n[N, minute]) for N in range(NMAX + 1)], w=[int(w[N, minute]) for N in range(NMAX + 1)],
                        ntr=int(ntr[0, minute]), wtr=int(wtr[0, minute]), nte=int(nte[0, minute]), wte=int(wte[0, minute])))
    return out


def live_run(ts, d, side, k, minute, hs, hour):
    """Today's data only. Number of signals that lost IN A ROW right before the slot (hour, minute) of this rule.
    Returns None when it cannot be known (no candles for an earlier signal hour at all)."""
    if len(ts) == 0:
        return None
    _, dirm, val = build_matrices(ts, d)
    W, V = outcome_cubes(dirm, val, side, k)
    w, v = W[-1], V[-1]
    run = 0
    seen = 0
    for h in range(hs, hour):
        if v[h, minute]:
            seen += 1
            run = 0 if w[h, minute] else run + 1
    if hour > hs and seen == 0:
        return None
    return run


def dedupe(rows):
    """rows sorted best first. Drops variants of the same underlying bug: same pair + direction, overlapping hour
    window and overlapping candle range [minute, minute+mtg] (e.g. ':30 No-MTG' and ':29 MTG1' are one bug)."""
    out = []
    for r in rows:
        a0, a1 = r["minute"], r["minute"] + r["mtg"]
        dup = False
        for s in out:
            if (s["pair"] == r["pair"] and s["side"] == r["side"] and r["hs"] < s["he"] and s["hs"] < r["he"]
                    and a0 <= s["minute"] + s["mtg"] and s["minute"] <= a1):
                dup = True
                break
        if not dup:
            out.append(r)
    return out

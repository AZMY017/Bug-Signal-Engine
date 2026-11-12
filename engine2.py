"""Engine 2 — Back Signal Discovery (pure numpy, no database, no Telegram). Separate from engine.py (engine.py is untouched).

Differences from engine 1 (on purpose):
 * Signal = every hour, every day: the (pair, minute-of-hour, direction) slot. Whole day only (UTC+6), all 60 minutes.
 * Back-to-back loss = consecutive SIGNAL OCCURRENCES in one continuous chronological series (07:30 loss, 08:30 loss ...),
   it does NOT restart each day. Missing candles (market closed / gaps) are skipped, they never count as a loss.
 * "After N losses" = next signal after EXACTLY N losses in a row (N = 1..5): total cases, wins, recovery accuracy.
 * MTG0 / MTG1 (MTG2 optional): normal win, MTG win, final win, final loss, MTG usage, normal vs final accuracy.
 * Doji (open == close) is never green or red: it is counted and reported separately (never a win).
 * Discovery on the FIRST part of the days only (train). The shown/validated accuracy is the LAST part (out of sample).
 * Monthly stability + 3-fold walk-forward consistency + multiple-testing correction (Bonferroni z) + payout based EV.
"""
import math
from statistics import NormalDist

import numpy as np

import engine

NREC = 5           # recovery after exactly 1..5 losses
MIN_MONTH_N = 15   # a month counts only with at least this many signals


def bonferroni_z(n_tests, alpha=0.05):
    """One-sided z so that the chance of even ONE fake pass among n_tests random slots stays about alpha."""
    n_tests = max(1, n_tests)
    return NormalDist().inv_cdf(1.0 - alpha / n_tests)


def ev_per_signal(nw, mw, lo, payout, mult):
    """Expected profit per signal in units of the first stake.
    Normal win: +payout. MTG win: -1 + mult*payout. Final loss: -(1 + mult) (MTG) / -1 (no MTG, mult=0)."""
    n = nw + mw + lo
    if n <= 0:
        return 0.0
    if mw == 0 and mult == 0:
        return (nw * payout - lo) / n
    return (nw * payout + mw * (-1 + mult * payout) - lo * (1 + mult)) / n


def seq_pass(Wt, Vt):
    """Wt, Vt: (T,60) bool, rows = signal occurrences in time order. -> n[NREC+1,60], wins[NREC+1,60], maxrun[60]."""
    T, mm = Wt.shape
    run = np.zeros(mm, np.int64)
    n = np.zeros((NREC + 1, mm), np.int64)
    wins = np.zeros((NREC + 1, mm), np.int64)
    maxrun = np.zeros(mm, np.int64)
    ks = np.arange(1, NREC + 1)[:, None]
    for t in range(T):
        v = Vt[t]
        w = Wt[t]
        n[0] += v
        wins[0] += w
        eq = (run[None, :] == ks) & v[None, :]
        n[1:] += eq
        wins[1:] += eq & w[None, :]
        run = np.where(v, np.where(w, 0, run + 1), run)
        np.maximum(maxrun, run, out=maxrun)
    return n, wins, maxrun


def _month_ids(d0, nd):
    days = (d0 + np.arange(nd)).astype("datetime64[D]")
    m = days.astype("datetime64[M]").astype(np.int64)
    starts = np.concatenate(([0], np.nonzero(np.diff(m))[0] + 1))
    return starts


def scan_pair(pair, ts, d, p):
    """p: dict(min_acc, min_n, min_test_n, z, split, sides, mtgs, minutes, payout, mult).
    Returns ALL slot rows (every minute x side x mtg) with a 'status':
      valid = discovered on train, confirmed out of sample, stable;  low = passes train but too little out-of-sample data;
      overfit = passes train, fails out of sample / stability;  no = does not pass discovery."""
    if len(ts) < 5000:
        return [], None
    d0, dirm, val = engine.build_matrices(ts, d)
    nd = dirm.shape[0]
    cut = max(1, int(nd * p["split"]))
    starts = _month_ids(d0, nd)
    f1, f2 = nd // 3, 2 * (nd // 3)
    z = p["z"]
    full = val[:, :1440].sum(1) >= 600          # "full" trading days (weekend edge days with a few candles are ignored)
    open_days = int(full.sum())
    quality = dict(days=open_days, coverage=float(val[full, :1440].sum()) / max(1, open_days * 1440),
                   doji=float(((dirm[:, :1440] == 0) & val[:, :1440]).sum()) / max(1, int(val[:, :1440].sum())))
    dj_cube = ((dirm[:, :1440] == 0) & val[:, :1440]).reshape(nd, 24, 60)
    rows = []
    minutes = p["minutes"]
    for side, sname in ((1, "UP"), (-1, "DOWN")):
        if sname not in p["sides"]:
            continue
        W0, _ = engine.outcome_cubes(dirm, val, side, 0)
        for k in p["mtgs"]:
            W, V = engine.outcome_cubes(dirm, val, side, k)
            base = engine.base_rate(k)
            Wt = W.reshape(nd * 24, 60)
            Vt = V.reshape(nd * 24, 60)
            cutr = cut * 24
            ntr, wtr, mtr = seq_pass(Wt[:cutr], Vt[:cutr])
            nte, wte, mte = seq_pass(Wt[cutr:], Vt[cutr:])
            n, w = ntr + nte, wtr + wte
            maxl = np.maximum(mtr, mte)
            normal_w = (W0 & V).sum((0, 1))
            doji_n = (dj_cube & V).sum((0, 1))
            dw, dn = W.sum(1), V.sum(1)                        # (D,60) per-day wins / signals
            mw_, mn_ = np.add.reduceat(dw, starts, axis=0), np.add.reduceat(dn, starts, axis=0)
            fw = np.stack([dw[:f1].sum(0), dw[f1:f2].sum(0), dw[f2:].sum(0)])
            fn = np.stack([dn[:f1].sum(0), dn[f1:f2].sum(0), dn[f2:].sum(0)])
            for m in minutes:
                a, b = int(ntr[0, m]), int(wtr[0, m])
                if int(n[0, m]) == 0:
                    continue
                lb = engine.wilson_lb(b, a, z)
                train_ok = a >= p["min_n"] and b / a >= p["min_acc"] and lb > base
                c, e = int(nte[0, m]), int(wte[0, m])
                okm = mn_[:, m] >= MIN_MONTH_N
                mwr = np.where(okm, mw_[:, m] / np.maximum(mn_[:, m], 1), 0.0)
                months_tot = int(okm.sum())
                months_ok = int(((mwr > base) & okm).sum())
                fold_ok = bool(all(fn[i, m] > 0 and fw[i, m] / fn[i, m] > base for i in range(3)))
                if not train_ok:
                    status = "no"
                elif c < p["min_test_n"]:
                    status = "low"
                elif (e / c >= p["min_acc"] - 0.05 and e / c >= base + 0.02 and fold_ok
                      and months_tot > 0 and months_ok >= 0.7 * months_tot):
                    status = "valid"
                else:
                    status = "overfit"
                nn, ww, nw = int(n[0, m]), int(w[0, m]), int(normal_w[m])
                mtg_w = ww - nw
                loss = nn - ww
                rows.append(dict(
                    pair=pair, minute=m, side=sname, mtg=k, hs=0, he=24, n=nn, w=ww, nw=nw, mw=mtg_w, loss=loss,
                    doji=int(doji_n[m]), ntr=a, wtr=b, nte=c, wte=e, maxl=int(maxl[m]),
                    rec_n=[int(n[i, m]) for i in range(1, NREC + 1)], rec_w=[int(w[i, m]) for i in range(1, NREC + 1)],
                    months_ok=months_ok, months_tot=months_tot, fold_ok=fold_ok, base=base,
                    score=lb - base, status=status, ev=ev_per_signal(nw, mtg_w, loss, p["payout"], p["mult"] if k else 0.0)))
    return rows, quality


STATUS_ORDER = {"valid": 3, "low": 2, "overfit": 1, "no": 0}


def sort_rows(rows):
    rows.sort(key=lambda r: (STATUS_ORDER[r["status"]], r["score"]), reverse=True)
    return rows

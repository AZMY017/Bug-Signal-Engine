"""Pair lists + which Neon database holds which pair. MUST stay identical to the live bot (signals.py + store._shard)."""
import datetime as dt
import os

BDT = dt.timezone(dt.timedelta(hours=6))  # UTC+6
# Frozen original order of the live bot: the candle database stores a pair by its position here. NEVER edit.
LEGACY_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "EUR_GBP", "EUR_JPY", "GBP_JPY", "AUD_USD", "NZD_USD",
                "USD_CAD", "USD_CHF", "AUD_JPY", "CAD_JPY", "CHF_JPY", "NZD_JPY", "EUR_AUD", "EUR_CAD",
                "EUR_CHF", "GBP_AUD", "GBP_CAD", "GBP_CHF", "AUD_CAD", "AUD_CHF", "AUD_NZD", "NZD_CAD",
                "NZD_CHF", "CAD_CHF"]
PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "EUR_GBP", "EUR_JPY", "GBP_JPY", "AUD_USD",
         "USD_CAD", "USD_CHF", "AUD_JPY", "CAD_JPY", "CHF_JPY",
         "EUR_AUD", "EUR_CAD", "EUR_CHF", "GBP_AUD", "GBP_CAD", "GBP_CHF", "AUD_CAD", "AUD_CHF"]
PID = {p: i for i, p in enumerate(LEGACY_PAIRS)}
_ACTIVE = {p: i for i, p in enumerate(PAIRS)}


def shard(pair, n_db):
    if pair in _ACTIVE:
        return _ACTIVE[pair] % n_db
    return PID[pair] % n_db


# ---------------------------------------------------------------- OTC market (candles collected by the OTC-MARKET-FUTURE-SIGNAL-BOT)
# Same list/order as that bot. More pairs: OTC_EXTRA_PAIRS=EURUSD_otc,GBPUSD_otc (exactly as that bot names them).
OTC_PAIRS = [
    "AUDNZD_otc", "BRLUSD_otc", "CADCHF_otc", "EURNZD_otc", "GBPNZD_otc", "NZDCAD_otc",
    "NZDCHF_otc", "NZDJPY_otc", "NZDUSD_otc", "UKBrent_otc", "USCrude_otc", "USDARS_otc",
    "USDBDT_otc", "USDCOP_otc", "USDDZD_otc", "USDEGP_otc", "USDIDR_otc", "USDINR_otc",
    "USDMXN_otc", "USDNGN_otc", "USDPHP_otc", "USDPKR_otc", "USDZAR_otc",
]
for _p in os.environ.get("OTC_EXTRA_PAIRS", "").replace(" ", "").split(","):
    if _p and _p not in OTC_PAIRS:
        OTC_PAIRS.append(_p)
_OTC_KEY = {}
for _p in OTC_PAIRS:
    _k = "".join(ch for ch in _p.upper() if ch.isalpha())          # 'AUDNZDOTC'
    _OTC_KEY[_k] = _p
    _OTC_KEY[_k[:-3]] = _p                                          # 'AUDNZD' (only used when it is not a real pair)


def is_otc(p):
    return p.endswith("_otc")


def plist(mkt):
    """Pair list of a market: 'otc' -> OTC pairs, anything else -> real-market pairs."""
    return OTC_PAIRS if mkt == "otc" else PAIRS


def pretty(p):
    if is_otc(p):
        return p[:-4] + "-OTC"
    return p.replace("_", "/")


def norm(text):
    """'eurusd' / 'EUR/USD' / 'eur_usd' -> 'EUR_USD'; 'audnzd_otc' / 'AUDNZD-OTC' -> 'AUDNZD_otc' (None if unknown)."""
    t = "".join(ch for ch in text.upper() if ch.isalpha())
    if len(t) == 6:
        p = t[:3] + "_" + t[3:]
        if p in PAIRS:
            return p
    return _OTC_KEY.get(t)

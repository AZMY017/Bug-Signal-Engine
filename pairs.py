"""Pair lists + which Neon database holds which pair. MUST stay identical to the live bot (signals.py + store._shard)."""
import datetime as dt

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


def pretty(p):
    return p.replace("_", "/")


def norm(text):
    """'eurusd' / 'EUR/USD' / 'eur_usd' -> 'EUR_USD' (None if unknown)."""
    t = "".join(ch for ch in text.upper() if ch.isalpha())
    if len(t) == 6:
        p = t[:3] + "_" + t[3:]
        if p in PAIRS:
            return p
    return None

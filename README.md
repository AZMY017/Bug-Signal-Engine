# Bug Signal Engine (আলাদা বট)

লাইভ বটের Neon ক্যান্ডেল ডেটা (শুধু পড়া) স্ক্যান করে সময়ভিত্তিক "বাগ সিগন্যাল" খোঁজে, নিজের আলাদা Neon ডেটাবেজে রেজাল্ট রাখে, আর চাইলে লাইভ অ্যালার্ট দেয়।
পুরো সেটআপ আর ব্যবহারের নিয়ম `GUIDE_bangla.md`-তে।

ফাইল: `bot.py` (টেলিগ্রাম + স্কেজুল) • `engine.py` (স্ক্যানের হিসাব) • `source.py` (Neon থেকে read-only পড়া) •
`resultsdb.py` (ইঞ্জিনের নিজের ডেটাবেজ) • `live.py` (লাইভ অ্যালার্ট) • `engine3.py` (ইঞ্জিন ৩: মিনিট গ্রিড, PDF/CSV) • `pairs.py` (পেয়ার/ডেটাবেজ ভাগ, লাইভ বটের সাথে হুবহু এক)

Render: Build `pip install -r requirements.txt` • Start `python bot.py`


## OTC market
Engines 3/4/6/8/9/10 have a Real / OTC switch. OTC candles are read (read-only) from the OTC-MARKET-FUTURE-SIGNAL-BOT databases:
set `OTC_DATABASE_URL_1/2/3` (same links, same order as that bot's `DATABASE_URL_1/2/3`). OTC has no live candles, so the live radar alerts stay real-market only.

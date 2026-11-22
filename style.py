"""Message styling for the settings / intro screens of the engines (/engine2 ... /engine13).

fancy(plain_text) -> Telegram HTML (send it with parse_mode="HTML"):
  * title line bold, coloured emoji bars (green / blue / orange) between the sections,
  * the explanation and ℹ️ notes in a quote block, "• key: value" lines with an icon and a bold value,
  * the ▶️ চালাও চাপো hint and the market line highlighted.
Only the look changes: the words, numbers and buttons come unchanged from the engine's own text.
"""
import html
import re

GREEN, BLUE, ORANGE = "🟩" * 9, "🟦" * 9, "🟧" * 9
ICONS = (("টাইম ফ্রেম", "⏱"), ("সময়কাল", "🗓"), ("টেস্টের দিন", "📅"), ("আউটপুট", "📤"), ("MTG", "🔁"), ("পেয়ার", "💱"),
         ("মিনিট", "🕒"), ("ব্যাক-টু-ব্যাক", "📉"), ("গণনা", "🔢"), ("ডিরেকশন", "↕️"), ("দিক", "↕️"), ("ন্যূনতম অ্যাকুরেসি", "🎯"),
         ("ন্যূনতম নমুনা", "🧪"), ("ন্যূনতম সিগন্যাল", "🔢"), ("ডোজি", "⚪"), ("মেসেজ", "💬"), ("সাথে পাঠাবে", "📎"),
         ("EMA", "📈"), ("ট্রেন্ড", "🧭"), ("তারিখ", "📅"), ("রাডার", "📡"))
_KV = re.compile(r"^• ([^:]{1,60}): (.*)$")


def _e(s):
    return html.escape(s, quote=False)


def _icon(key):
    for k, ic in ICONS:
        if k in key:
            return ic
    return "▫️"


def _bullet(line):
    m = _KV.match(line)
    if not m:
        return _e(line)
    return "%s %s: <b>%s</b>" % (_icon(m.group(1)), _e(m.group(1)), _e(m.group(2)))


def _lines(block, quote=False):
    out = []
    for ln in block.split("\n"):
        if ln.startswith("• "):
            out.append(_bullet(ln))
        elif ln.startswith("(") and ln.endswith(")"):
            out.append("<i>%s</i>" % _e(ln))
        else:
            out.append(_e(ln))
    return "\n".join(out)


def fancy(text):
    blocks = [b for b in (text or "").strip().split("\n\n") if b.strip()]
    if not blocks:
        return ""
    out = ["<b>%s</b>" % _e(blocks[0].split("\n")[0])]
    rest0 = blocks[0].split("\n")[1:]
    if rest0:
        out.append(_lines("\n".join(rest0)))
    out.append(GREEN)
    blue_done = orange_done = False
    for b in blocks[1:]:
        first = b.split("\n")[0]
        if first.startswith("⚙️"):
            if not blue_done:
                out.append(BLUE)
                blue_done = True
            lines = b.split("\n")
            head = lines[0].rstrip(":")
            out.append("<b>%s</b>" % _e(head))
            out.append(_lines("\n".join(lines[1:])))
        elif first.startswith(("🌍 মার্কেট", "🧪 মার্কেট")):
            if not orange_done:
                out.append(ORANGE)
                orange_done = True
            lines = b.split("\n")
            m = re.match(r"^(\S+ মার্কেট): (.*)$", lines[0])
            out.append("%s: <b>%s</b>" % (_e(m.group(1)), _e(m.group(2))) if m else _e(lines[0]))
            if len(lines) > 1:
                out.append(_lines("\n".join(lines[1:])))
        elif first.startswith("ℹ️"):
            lines = b.split("\n")
            out.append("<blockquote>%s\n%s</blockquote>" % ("<b>%s</b>" % _e(lines[0]), _lines("\n".join(lines[1:]))))
        elif first.startswith("⏳"):
            out.append("<b>%s</b>" % _e(b))
        elif "চালাও চাপো" in b:
            out.append("👉 " + _e(b).replace("চালাও চাপো", "<b>চালাও চাপো</b>"))
        elif first.startswith(("📡", "🧪", "🌍")) and "\n" not in b and len(b) < 220:
            out.append(_e(b))
        elif first.startswith("•") or first.startswith("("):
            out.append(_lines(b))
        else:
            out.append("<blockquote>%s</blockquote>" % _lines(b))
    return "\n".join(out)

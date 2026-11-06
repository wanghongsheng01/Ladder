"""Rule-based de-identification of RMA customer-service memos before they are sent to any external LLM API.

Replaces personal and transactional identifiers with typed placeholders and keeps the text that describes the
return reason (product, defect, shipment problem, customer intent). Runs fully offline; no text leaves the machine.

Masked (placeholder):
  [EMAIL] e-mail addresses            [URL]   URLs and domains
  [PHONE] phone numbers               [ID]    order / tracking / RMA / account numbers (>= 6 digits, or long codes)
  [ADDR]  street addresses, PO boxes, apartment/suite/unit parts, postcodes, Chinese addresses
  [NAME]  person names after titles/labels (Mr, Ms, Herr, Frau, name:, cx:, 先生/女士 ...) and capitalised name pairs
          that are not ordinary dictionary words
  [CARD]  payment-card-like digit groups   [IP] IP addresses
Kept: product model tokens with < 6 digits (e.g. "E7", "BS2-1"), amounts, dates, ordinary words.

Usage:
  python deidentify.py IN.parquet OUT.parquet [--col memo]      # writes memo_deid + counts per placeholder
  python deidentify.py --selftest
"""
import os, re, sys

# ---------------------------------------------------------------- vocabulary for the name heuristic
_WORDS = set()
for p in ["/usr/share/dict/words", "/usr/dict/words"]:
    if os.path.exists(p):
        with open(p, errors="ignore") as f:
            _WORDS = {w.strip() for w in f if w.strip() and w.strip()[0].islower()}  # common (lower-case) words only
        break
_DOMAIN = {"flexispot", "loctek", "amazon", "ebay", "wayfair", "walmart", "fedex", "ups", "usps", "dhl", "dpd", "gls",
           "hermes", "tnt", "royal", "mail", "parcelforce", "evri", "yodel", "canada", "post", "australia", "germany",
           "france", "spain", "italy", "netherlands", "poland", "uk", "us", "usa", "eu", "fba", "fbm", "sku", "asin",
           "paypal", "stripe", "shopify", "aliexpress", "otto", "cdiscount", "bol", "kaufland", "mediamarkt", "home",
           "depot", "office", "costco", "target", "best", "buy", "desk", "chair", "frame", "top", "table", "monitor",
           "arm", "keypad", "controller", "motor", "leg", "legs", "box", "carton", "label", "refund", "return",
           "replacement", "warranty", "customer", "cx", "tabletop", "tabletops", "desktop", "standing", "sit", "stand",
           "rts", "rma", "trk", "ts", "oos", "cn", "team", "claim", "case", "walmart", "amazon's", "costco", "uline", "pro", "plus", "max", "mini", "lite", "comhar", "kana",
           "e7", "e5", "e6", "e8", "e9", "e1", "e2", "eg", "ef", "en", "q8", "cb", "ss", "ok", "pls", "please",
           "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january", "february",
           "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"}

_FIRST = set()
if os.path.exists("/usr/share/dict/propernames"):
    with open("/usr/share/dict/propernames") as f:
        _FIRST = {w.strip() for w in f if w.strip()}
# first names that are also frequent capitalised English words in memos
_FIRST -= {"Will", "May", "Mark", "Bill", "Frank", "Grant", "Sue", "Rob", "Art", "Don", "Guy", "Ray", "Hope", "Joy",
           "Faith", "Dawn", "Rose", "June", "April", "August", "Case", "Price", "Jack", "Win", "Cliff", "Glen",
           "Gene", "Lane", "Dale", "Hall", "Max", "Pro"}
# drop proper names that are also ordinary lower-case words ("The", "No", "Part", "Hall", ...)
_FIRST = {n for n in _FIRST if n.lower() not in _WORDS}

# ---------------------------------------------------------------- patterns (order matters)
EMAIL = re.compile(r"[\w.+-]+\s?@\s?[\w-]+(\.[\w-]+)+|[\w.+-]+\s?(\(at\)|\[at\]|\sat\s)\s?[\w-]+\s?(\.|dot)\s?(com|net|org|de|co\.uk|fr)\b", re.I)
URL = re.compile(r"\b(https?://\S+|www\.\S+|\b[\w-]+\.(com|net|org|de|co\.uk|fr|es|it|nl|pl|jp|cn)(/\S*)?\b)", re.I)
IPADDR = re.compile(r"\b\d{1,3}(\.\d{1,3}){3}\b")
CARD = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")
PHONE = re.compile(r"(\+?\d{1,3}[\s.-]?)?\(\d{2,4}\)[\s.-]?\d{3,4}[\s.-]?\d{3,5}\b"
                   r"|(?<![\w/])(\+?\d{1,3}[\s.-]?)?(\(\d{2,4}\)[\s.-]?)?\d{2,4}[\s.-]\d{2,4}[\s.-]\d{2,5}(?![\w/])"
                   r"|(?<![\w/])\+\d{7,15}\b|(?<![\w/-])\(?0\d{2,4}\)?[\s/-]?\d{5,9}\b")
LONG_ID = re.compile(r"\b(?=[A-Z0-9-]*\d{6,})[A-Z0-9][A-Z0-9-]{5,}\b|\b\d{3}-\d{7}-\d{7}\b|\b1Z[0-9A-Z]{16}\b", re.I)
# street addresses (EN / DE / FR / ES / IT / NL / PL), PO boxes, unit designators, postcodes
SUFFIX = (r"(street|st|road|rd|avenue|ave|av|boulevard|blvd|drive|dr|lane|ln|way|court|ct|place|pl|parkway|pkwy|"
          r"highway|hwy|terrace|ter|circle|cir|crescent|close|square|sq|trail|trl|row|walk|grove|gardens|"
          r"straße|strasse|str|weg|allee|platz|gasse|ring|damm|ufer|chaussee|rue|chemin|avenida|calle|via|viale|"
          r"piazza|straat|laan|plein|ulica|ul)")
STREET = re.compile(r"\b\d{1,6}[a-z]?\s+(?:[\w'.-]+\s+){0,4}" + SUFFIX + r"\b\.?"
                    r"|\b[\wäöüß'.-]+" + r"(straße|strasse|str\.|weg|allee|platz|gasse|ring|damm)\s*\d{1,5}[a-z]?\b"
                    r"|\b(rue|avenue|chemin|calle|avenida|via|viale|piazza|ul\.?|ulica)\s+(?:[\w'.-]+\s+){0,4}\d{1,5}\b"
                    r"|\b" + SUFFIX + r"\s+\d{1,5}[a-z]?\b", re.I)
POBOX = re.compile(r"\b(p\.?\s?o\.?\s?box|postfach|apartado)\s*\d+\b", re.I)
UNIT = re.compile(r"\b(apt|apartment|suite|ste|unit|flat|floor|fl|room|rm|bldg|building)\.?\s*#?\s*[a-z]?\d[a-z0-9-]{0,5}\b", re.I)
POSTCODE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b"            # UK
                      r"|\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b"                    # CA
                      r"|\b\d{5}(-\d{4})?\b"                              # US / DE / FR / ES / IT
                      r"|\b\d{4}\s?[A-Z]{2}\b"                            # NL
                      r"|\b\d{2}-\d{3}\b")                                # PL
CN_ADDR = re.compile(r"[一-龥]{2,}(省|市|区|县|镇|乡|村|街道|路|街|巷|弄|号|栋|幢|单元|室|楼|小区|大厦|广场)"
                     r"([一-龥\d\-#]{0,30}(号|室|楼|栋|幢|单元))?")
# names
TITLE_NAME = re.compile(r"\b(mr|mrs|ms|miss|dr|herr|frau|mme|mlle|m|sr|sra|sig|signor|signora|pan|pani)\.?\s+"
                        r"([A-Z][\w'-]+)(\s+[A-Z][\w'-]+)?", re.I)
LABEL_NAME = re.compile(r"\b(name|customer name|recipient|receiver|contact|cx name|buyer|consignee|attn|attention|"
                        r"ship to|deliver to|bill to)\s*[:：-]\s*([^\n,;|]{1,60})", re.I)
CN_NAME = re.compile(r"(?:联系人|收件人|姓名|客户)[:：]?\s*[一-龥]{2,4}|[一-龥]{1,2}(先生|女士|小姐)")
FIRSTNAME = re.compile(r"\b(" + "|".join(sorted(map(re.escape, _FIRST), key=len, reverse=True)) + r")\b(\s+[A-Z][\w'-]+)?") if _FIRST else None
CX_NAME = re.compile(r"\b(cx|customer|client|buyer|recipient|caller)\s+(?:is\s+|named\s+)?([A-Z][a-z'-]+)(\s+[A-Z][a-z'-]+)?")
CITY_AFTER = re.compile(r"\[ADDR\][,\s]+(?:[A-Z][\wäöüéè'-]+[\s,]+){1,3}(?:[A-Z]{2}\b)?|\[ADDR\]\s+[A-ZÄÖÜ][\wäöüß-]+")
CITY_BEFORE = re.compile(r"\b(?:[A-Z][a-z]+\s){1,3}[A-Z]{2}\s*,?\s*\[ADDR\]")
# staff names in workflow notes: "approved by TL Abi", "check By ATL X", "sent to X", trailing "- X" signatures
ROLE = r"(?:(?:TL|ATL|SV|TM|QA|CS|Sup|Supervisor|Lead|Agent|Manager|Mgr)\.?\s+)"
STAFF = re.compile(r"\b((?:approved|checked|check|confirmed|handled|escalated|signed|reviewed)\s+by|by|sent\s+to|per|cc|from|thanks?|regards)"
                   r"\s+" + ROLE + r"?([A-Z][a-z]{2,})\b|\b" + ROLE + r"([A-Z][a-z]{2,})\b", re.I)
SIGN = re.compile(r"[-–—~]\s*([A-Z][a-z]{2,})\s*\.?\s*$")
CAP_PAIR = re.compile(r"\b([A-Z][a-z]{1,15})\s+([A-Z][a-z]{1,20}(?:-[A-Z][a-z]+)?)\b")

EMAIL_ANY = re.compile(r"[\w.+-]+@[\w-]+(?:[.,]\s?[\w-]+)*|\S+@\S+")  # typos such as "name@yahoo,com"
ORDER = [("EMAIL", EMAIL), ("EMAIL", EMAIL_ANY), ("URL", URL), ("IP", IPADDR), ("CARD", CARD), ("ID", LONG_ID), ("PHONE", PHONE),
         ("ADDR", POBOX), ("ADDR", STREET), ("ADDR", CN_ADDR), ("NAME", TITLE_NAME), ("NAME", LABEL_NAME),
         ("NAME", CN_NAME), ("ADDR", UNIT), ("ADDR", POSTCODE)]


def _common(w):
    w = w.lower()
    if w in _WORDS or w in _DOMAIN: return True
    for suf, add in [("s", ""), ("es", ""), ("ed", ""), ("ed", "e"), ("ing", ""), ("ing", "e"), ("ly", ""), ("er", "")]:
        if w.endswith(suf) and (w[: -len(suf)] + add) in _WORDS: return True
    return False


def _is_name_pair(m):
    a, b = m.group(1), m.group(2).split("-")[0]
    # an ordinary word on either side ("Table Top", "Damaged Tabletop") -> not a name; a known first name -> name
    if a in _FIRST: return not _common(b) or b[0].isupper()
    return not _common(a) and not _common(b)


TOKEN = re.compile(r"\b[^\W\d_]{3,}\b")
_RARE = set()


def build_rare_vocab(texts, k=5):
    """Corpus-level k-anonymity on tokens: alphabetic tokens (any case) that occur in fewer than k memos of the
    whole corpus and are not ordinary dictionary/domain words are treated as identifying (mostly person, street
    and place names; also rare typos, whose loss is harmless)."""
    from collections import Counter
    df = Counter()
    for t in texts:
        if isinstance(t, str):
            df.update({w.lower() for w in TOKEN.findall(t)})
    return {w for w, n in df.items() if n < k and not _common(w)}


def deidentify(text):
    """Return (masked_text, counts)."""
    if not isinstance(text, str) or not text:
        return text, {}
    counts = {}
    s = text
    for tag, pat in ORDER:
        s, n = pat.subn(f"[{tag}]", s)
        if n: counts[tag] = counts.get(tag, 0) + n
    def _staff(m):
        name = m.group(2) or m.group(3)
        return m.group(0) if _common(name) else m.group(0)[: m.group(0).rfind(name)] + "[NAME]"
    s = STAFF.sub(_staff, s)
    s = SIGN.sub(lambda m: m.group(0) if _common(m.group(1)) else m.group(0).replace(m.group(1), "[NAME]"), s)
    s = CX_NAME.sub(lambda m: m.group(1) + " [NAME]" if m.group(2).lower() not in _DOMAIN and m.group(2).lower() not in _WORDS else m.group(0), s)
    if FIRSTNAME is not None:
        s = FIRSTNAME.sub(lambda m: "[NAME]" if m.group(1).lower() not in _DOMAIN else m.group(0), s)
    s, n = CAP_PAIR.subn(lambda m: "[NAME]" if _is_name_pair(m) else m.group(0), s)
    s = CITY_BEFORE.sub("[ADDR]", s); s = CITY_AFTER.sub("[ADDR]", s)
    if _RARE:
        s = TOKEN.sub(lambda m: "[NAME]" if m.group(0).lower() in _RARE else m.group(0), s)
    n = s.count("[NAME]") - counts.get("NAME", 0)
    if n > 0: counts["NAME"] = counts.get("NAME", 0) + n
    s = re.sub(r"(\[(ADDR|NAME|PHONE|ID)\][\s,.;:-]*){2,}", lambda m: re.findall(r"\[\w+\]", m.group(0))[0] + " ", s)
    return s.strip(), counts


RESIDUAL = {"email": EMAIL, "at_sign_word": re.compile(r"[A-Za-z0-9]@[A-Za-z]"), "long_digits": re.compile(r"\d{6,}"), "phone_like": re.compile(r"\d{3}[\s.-]\d{3}[\s.-]\d{3,4}"),
            "street_like": STREET, "title_name": re.compile(r"\b(mr|mrs|ms|herr|frau)\.?\s+[A-Z][a-z]+", re.I)}


def residual_scan(texts):
    """Count de-identified texts that still match an identifier pattern (should all be 0)."""
    return {k: int(sum(bool(p.search(t)) for t in texts if isinstance(t, str))) for k, p in RESIDUAL.items()}


def _selftest():
    cases = ["Cx John Smith at 1234 Maple Street Apt 5B, Springfield IL 62704 wants refund, call +1 (217) 555-0134",
             "Order 114-1234567-1234567 tracking 1Z999AA10123456784, email jane.doe@example.com, desk top damaged",
             "Herr Müller, Hauptstraße 12, 10115 Berlin: Motor defekt, E7 Pro Tischplatte zerkratzt",
             "Ms. Garcia says keypad shows E01 error; replacement controller sent via UPS",
             "收件人 张伟 北京市朝阳区建国路88号 电话 13800138000 桌腿缺少螺丝",
             "parts missing for E5 frame, cx changed mind, partial refund 30 USD to keep"]
    for c in cases:
        print(deidentify(c)[0])
    out = [deidentify(c)[0] for c in cases]
    print("residual", residual_scan(out))


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest(); sys.exit(0)
    import pandas as pd
    src, dst = sys.argv[1], sys.argv[2]
    col = sys.argv[sys.argv.index("--col") + 1] if "--col" in sys.argv else "memo"
    df = pd.read_parquet(src)
    if "--rare-from" in sys.argv:  # corpus used for token document frequencies (the full memo table)
        corpus = pd.read_parquet(sys.argv[sys.argv.index("--rare-from") + 1])[col]
        _RARE.update(build_rare_vocab(corpus))
        print("rare-token vocabulary", len(_RARE))
    res = df[col].map(deidentify)
    df[col + "_deid"] = res.map(lambda r: r[0])
    for tag in ["EMAIL", "URL", "IP", "CARD", "ID", "PHONE", "ADDR", "NAME"]:
        df["n_" + tag] = res.map(lambda r: r[1].get(tag, 0))
    df.drop(columns=[col]).to_parquet(dst, index=False)
    print("rows", len(df), "rows with any mask %.3f" % (df[[c for c in df if c.startswith("n_")]].sum(1) > 0).mean())
    print({c: int(df[c].sum()) for c in df if c.startswith("n_")})
    print("residual", residual_scan(df[col + "_deid"].tolist()))

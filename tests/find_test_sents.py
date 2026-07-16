import re, pathlib, spacy

nlp = spacy.load("en_core_web_sm")
SUPPLY_PATS = re.compile(
    r"(suppli|customer|purchas|source|partner|subsidiar|compet|manufactur|depend|acqui)",
    re.IGNORECASE,
)

for ticker in ["LRCX", "NVDA", "AMD", "AVGO", "QCOM", "MU"]:
    text = pathlib.Path(f"data/filings/{ticker}_2023.txt").read_text(encoding="utf-8")
    text = re.sub(r"=== ITEM.*?===", "", text)
    sents = re.split(r"(?<=[a-z0-9])\.\s+(?=[A-Z])", text)
    for s in sents:
        s = s.strip()
        if not (60 < len(s) < 350 and SUPPLY_PATS.search(s)):
            continue
        doc = nlp(s)
        orgs = [e.text for e in doc.ents if e.label_ in ("ORG", "GPE", "PRODUCT")]
        if len(set(orgs)) >= 2:
            print(f"[{ticker}] orgs={orgs}")
            print(f"  {s[:200]}")
            print()

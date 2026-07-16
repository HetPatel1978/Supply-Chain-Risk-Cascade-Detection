"""
Download 10-K filings for a semiconductor supply-chain sample from SEC EDGAR,
extract Item 1 (Business) and Item 1A (Risk Factors), and save to data/filings/.

Supply chain graph (all US domestic filers, standard 10-K format):
  Equipment / memory upstream         Chip designers / integrators downstream
  ---------------------------------   ----------------------------------------
  LRCX  (Lam Research)                NVDA  (NVIDIA)
  AMAT  (Applied Materials)           AMD   (Advanced Micro Devices)
  KLAC  (KLA Corporation)             AVGO  (Broadcom)
  MU    (Micron Technology)           QCOM  (Qualcomm)

Notes:
  TSM / ASML: foreign private issuers filing 20-F, not 10-K.
  INTC: uses non-standard narrative format without Item N headers.
  Both replaced with standard US filers (MU, KLAC, AVGO).

SEC rate-limit guidance: <= 10 req/sec; script targets ~2 req/sec (0.5 s sleep).
User-Agent must identify requester per https://www.sec.gov/os/accessing-edgar-data
"""

import re
import time
import json
import pathlib
import requests
from html.parser import HTMLParser

OUT_DIR = pathlib.Path(__file__).parent.parent / "data" / "filings"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["LRCX", "AMAT", "KLAC", "MU", "NVDA", "AMD", "AVGO", "QCOM"]
FILING_YEAR = 2023

HEADERS = {
    "User-Agent": "Supply Chain Risk Research hetp2030@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

EDGAR_TICKERS_URL  = "https://www.sec.gov/files/company_tickers.json"
EDGAR_SUBMISSIONS  = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"

ITEM1_START  = re.compile(r"item\s*1[\.\:]\s*business",        re.IGNORECASE)
ITEM1A_START = re.compile(r"item\s*1a[\.\:]\s*risk\s*factors", re.IGNORECASE)
ITEM1B_START = re.compile(r"item\s*1b[\.\:]",                  re.IGNORECASE)
ITEM2_START  = re.compile(r"item\s*2[\.\:]\s*properties",      re.IGNORECASE)

# Fallback patterns for non-standard 10-K formats (e.g., Intel narrative style)
ITEM1_FALLBACK  = re.compile(r"introduction\s+to\s+our\s+business|overview\s+of\s+our\s+business", re.IGNORECASE)
ITEM1A_FALLBACK = re.compile(r"\brisk\s+factors\b",            re.IGNORECASE)
ITEM1A_END_FB   = re.compile(r"management.{0,10}discussion|quantitative.*qualitative|properties\b", re.IGNORECASE)


# ── helpers ──────────────────────────────────────────────────────────────────

class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
    def handle_data(self, data: str):
        self.parts.append(data)


def strip_html(html: str) -> str:
    # Remove iXBRL hidden header block (XBRL metadata, not visible content)
    html = re.sub(r"<ix:header\b[^>]*>.*?</ix:header>", "", html,
                  flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<div[^>]+display\s*:\s*none[^>]*>.*?</div>', "", html,
                  flags=re.DOTALL | re.IGNORECASE)
    p = _HTMLStripper()
    p.feed(html)
    text = " ".join(p.parts)
    return re.sub(r"\s+", " ", text).strip()


def get_all_ciks() -> dict[str, str]:
    r = requests.get(EDGAR_TICKERS_URL, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in r.json().values()}


def get_latest_10k(cik: str) -> tuple[str, str, str] | None:
    """Return (accession, filing_date, primary_doc_filename) for the most recent 10-K."""
    url = EDGAR_SUBMISSIONS.format(cik=cik)
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    recent = r.json().get("filings", {}).get("recent", {})
    forms   = recent.get("form", [])
    accnos  = recent.get("accessionNumber", [])
    dates   = recent.get("filingDate", [])
    pdocs   = recent.get("primaryDocument", [])

    rows = list(zip(forms, accnos, dates, pdocs))
    # prefer filing from target year; fall back to most recent 10-K
    for form, acc, date, pdoc in rows:
        if form == "10-K" and date.startswith(str(FILING_YEAR)):
            return acc, date, pdoc
    for form, acc, date, pdoc in rows:
        if form == "10-K":
            return acc, date, pdoc
    return None


def fetch_filing_text(cik: str, accession: str, primary_doc: str) -> str:
    acc_nodash = accession.replace("-", "")
    url = f"{EDGAR_ARCHIVE_BASE}/{int(cik)}/{acc_nodash}/{primary_doc}"
    r = requests.get(url, headers=HEADERS, timeout=90)
    r.raise_for_status()
    return strip_html(r.text) if "<html" in r.text.lower() else re.sub(r"\s+", " ", r.text).strip()


def _all_positions(pat: re.Pattern, text: str) -> list[int]:
    return [m.start() for m in pat.finditer(text)]


def extract_items(text: str) -> dict[str, str]:
    """
    10-Ks contain each item header twice: once in the compact Table of Contents
    and once at the actual section start.  We pick section boundaries by taking
    the LAST occurrence of each header before the next section's last occurrence.
    """
    i1_all  = _all_positions(ITEM1_START,  text)
    i1a_all = _all_positions(ITEM1A_START, text)
    i1b_all = _all_positions(ITEM1B_START, text)
    i2_all  = _all_positions(ITEM2_START,  text)

    # Real Item 1A = last occurrence that has > 500 chars before Item 1B / Item 2
    def real_section(positions: list[int], end_candidates: list[list[int]]) -> int:
        for pos in reversed(positions):
            nexts = [p for lst in end_candidates for p in lst if p > pos]
            if not nexts or (min(nexts) - pos) > 500:
                return pos
        return positions[-1] if positions else -1

    i1a = real_section(i1a_all, [i1b_all, i2_all])
    i1  = max((p for p in i1_all if p < i1a), default=-1) if i1a != -1 else (i1_all[-1] if i1_all else -1)
    i1b = next((p for p in sorted(i1b_all) if p > i1a), -1)
    i2  = next((p for p in sorted(i2_all)  if p > i1a), -1)

    if i1 != -1 and i1a != -1:
        item1 = text[i1:i1a].strip()
    elif i1 != -1:
        item1 = text[i1: i1 + 8_000].strip()
    else:
        item1 = ""

    if i1a != -1:
        end = next((p for p in sorted([i1b, i2]) if p > i1a and p != -1), -1)
        item1a = text[i1a: end].strip() if end != -1 else text[i1a: i1a + 25_000].strip()
    else:
        item1a = ""

    # Fallback for non-standard formats
    if len(item1) < 500 and len(item1a) < 500:
        fb1  = _all_positions(ITEM1_FALLBACK,  text)
        fb1a = _all_positions(ITEM1A_FALLBACK, text)
        fb_end = _all_positions(ITEM1A_END_FB, text)
        if fb1:
            fb1_pos = fb1[0]
            fb1a_pos = next((p for p in fb1a if p > fb1_pos + 1000), -1)
            if fb1a_pos != -1:
                item1  = text[fb1_pos:fb1a_pos].strip()
                end_fb = next((p for p in fb_end if p > fb1a_pos + 500), -1)
                item1a = text[fb1a_pos: end_fb].strip() if end_fb != -1 else text[fb1a_pos: fb1a_pos + 25_000].strip()

    return {"item1_business": item1, "item1a_risk_factors": item1a}


# ── main loop ────────────────────────────────────────────────────────────────

def process_ticker(ticker: str, cik: str) -> dict | None:
    print(f"\n[{ticker}] CIK={cik}")
    hit = get_latest_10k(cik)
    if hit is None:
        print(f"  No 10-K found — skipping.")
        return None
    accession, date, primary_doc = hit
    year = date[:4]
    print(f"  {accession}  ({date})  doc={primary_doc}")

    time.sleep(0.5)
    text = fetch_filing_text(cik, accession, primary_doc)
    print(f"  raw text: {len(text):,} chars")

    items = extract_items(text)
    out_path = OUT_DIR / f"{ticker}_{year}.txt"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("=== ITEM 1: BUSINESS ===\n\n")
        f.write(items["item1_business"] or "[Not extracted]")
        f.write("\n\n=== ITEM 1A: RISK FACTORS ===\n\n")
        f.write(items["item1a_risk_factors"] or "[Not extracted]")

    b_len  = len(items["item1_business"])
    ra_len = len(items["item1a_risk_factors"])
    print(f"  Saved {out_path.name}  (item1={b_len:,} | item1a={ra_len:,} chars)")
    time.sleep(0.5)
    return {"ticker": ticker, "year": year, "path": out_path, "items": items}


def main():
    print("Fetching EDGAR ticker->CIK map...")
    all_ciks = get_all_ciks()
    time.sleep(0.5)

    first = None
    for ticker in TICKERS:
        cik = all_ciks.get(ticker)
        if not cik:
            print(f"[{ticker}] not in EDGAR — skipping.")
            continue
        result = process_ticker(ticker, cik)
        if result and first is None:
            first = result

    if first:
        snippet = (first["items"]["item1_business"] or first["items"]["item1a_risk_factors"])[:500]
        t, y = first["ticker"], first["year"]
        print(f"\n{'='*60}")
        print(f"SANITY CHECK — {t}_{y}.txt  first 500 chars of Item 1:")
        print(f"{'='*60}")
        print(snippet)

    print("\nAll done. Files in data/filings/")


if __name__ == "__main__":
    main()

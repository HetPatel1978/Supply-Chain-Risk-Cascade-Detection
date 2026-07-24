"""
Clean and merge rule-based + Groq triples into one high-quality dataset.

Fixes:
  1. Resolve unresolved pronouns (<filing company>, we, us) using source_file
  2. Canonicalize company name variants (TSMC / Taiwan Semiconductor / etc.)
  3. Filter noisy / generic entities (bullet points, years, product names, etc.)
  4. Remove self-loops (head == tail after canonicalization)
  5. Deduplicate keeping highest confidence
"""

import json, pathlib, re

ROOT = pathlib.Path(__file__).parent.parent.parent / "results"

# ── Canonical company names ──────────────────────────────────────────────────
# Maps every known variant → one canonical name
CANON: dict[str, str] = {
    # TSMC
    "taiwan semiconductor manufacturing company limited": "TSMC",
    "taiwan semiconductor manufacturing company limited (tsmc)": "TSMC",
    "taiwan semiconductor manufacturing company": "TSMC",
    "taiwan semiconductor": "TSMC",
    "tsmc": "TSMC",
    # NVIDIA
    "nvidia corporation": "NVIDIA",
    "nvidia corp": "NVIDIA",
    "nvidia corp.": "NVIDIA",
    "nvidia": "NVIDIA",
    "nvda": "NVIDIA",
    # AMD
    "advanced micro devices, inc.": "AMD",
    "advanced micro devices, inc": "AMD",
    "advanced micro devices": "AMD",
    "amd": "AMD",
    # Applied Materials
    "applied materials, inc.": "Applied Materials",
    "applied materials, inc": "Applied Materials",
    "applied materials": "Applied Materials",
    "applied": "Applied Materials",
    # Lam Research
    "lam research corporation": "Lam Research",
    "lam research corp": "Lam Research",
    "lam research corp.": "Lam Research",
    "lam research": "Lam Research",
    "lam's": "Lam Research",
    "lam": "Lam Research",
    # KLA
    "kla corporation": "KLA",
    "kla corp": "KLA",
    "kla corp.": "KLA",
    "kla-tencor": "KLA",
    "kla's": "KLA",
    "kla": "KLA",
    # Micron
    "micron technology, inc.": "Micron",
    "micron technology, inc": "Micron",
    "micron technology": "Micron",
    "micron": "Micron",
    # Broadcom
    "broadcom corporation": "Broadcom",
    "broadcom corp.": "Broadcom",
    "broadcom corp": "Broadcom",
    "broadcom inc.": "Broadcom",
    "broadcom inc": "Broadcom",
    "broadcom": "Broadcom",
    # Qualcomm
    "qualcomm incorporated": "Qualcomm",
    "qualcomm inc.": "Qualcomm",
    "qualcomm inc": "Qualcomm",
    "qualcomm": "Qualcomm",
    # Intel
    "intel corporation": "Intel",
    "intel corp": "Intel",
    "intel corp.": "Intel",
    "intel": "Intel",
    # ASML
    "asml holding n.v.": "ASML",
    "asml holding": "ASML",
    "asml": "ASML",
    # Samsung
    "samsung electronics": "Samsung",
    "samsung": "Samsung",
    # Microsoft
    "microsoft corporation": "Microsoft",
    "microsoft": "Microsoft",
    # Sony
    "sony corporation": "Sony",
    "sony": "Sony",
    # SoftBank
    "softbank group corp": "SoftBank",
    "softbank": "SoftBank",
    # Arm
    "arm limited": "Arm",
    "arm holdings": "Arm",
    "arm": "Arm",
    # GLOBALFOUNDRIES
    "globalfoundries inc.": "GLOBALFOUNDRIES",
    "globalfoundries inc. (gf)": "GLOBALFOUNDRIES",
    "globalfoundries": "GLOBALFOUNDRIES",
    # UMC
    "united microelectronics corporation": "UMC",
    "umc": "UMC",
    # VMware
    "vmware": "VMware",
    # Xilinx
    "xilinx, inc.": "Xilinx",
    "xilinx": "Xilinx",
    # Dell
    "dell": "Dell",
    # HP / HPE
    "hewlett packard enterprise": "HPE",
    "hewlett-packard corp": "HP",
    # Tokyo Electron
    "tokyo electron, ltd.": "Tokyo Electron",
    "tokyo electron": "Tokyo Electron",
    # Screen Holdings
    "screen holding co., ltd.": "Screen Holdings",
    # Onto Innovation
    "onto innovation, inc.": "Onto Innovation",
    # Lasertec
    "lasertec, inc.": "Lasertec",
    # Hitachi
    "hitachi high-technologies corporation": "Hitachi",
    # Flex
    "flex ltd.": "Flex",
    # Amkor
    "amkor technology": "Amkor Technology",
    # Tongfu
    "tongfu microelectronics co., ltd": "Tongfu Microelectronics",
    # ASM International
    "asm international": "ASM International",
}

# Ticker → filer company name (for resolving pronouns)
TICKER_TO_COMPANY: dict[str, str] = {
    "LRCX": "Lam Research",
    "AMAT": "Applied Materials",
    "KLAC": "KLA",
    "MU":   "Micron",
    "NVDA": "NVIDIA",
    "AMD":  "AMD",
    "AVGO": "Broadcom",
    "QCOM": "Qualcomm",
    "INTC": "Intel",
}

# Generic / non-company terms to reject outright
_NOISE = {
    "we", "us", "our", "the company", "company", "the corporation",
    "customers", "customer", "suppliers", "supplier", "vendors", "vendor",
    "foundries", "foundry", "partners", "partner", "competitors", "competitor",
    "third parties", "third party", "third-party", "a third party",
    "manufacturers", "manufacturer", "third-party manufacturers",
    "third-party suppliers", "third-party providers",
    "original device manufacturers", "odms", "oems", "oem",
    "industry leaders", "start-ups", "diversified companies",
    "governmental entities", "critical infrastructure operators",
    "internet and csps", "mapping companies",
    "employees", "consultants", "retailers/distributors", "system builders",
    "tier-1 automotive suppliers", "automotive manufacturers",
    "semiconductor manufacturers", "our customers", "our suppliers",
    "outsource providers", "other service providers",
    "foundry/logic customers", "providers of semiconductor-based high-performance interconnect products based on infiniBand, ethernet, fibre channel and proprietary technologies",
    "companies that provide or intend to provide gpus, cpus, dpus, embedded socs, and other accelerated, ai computing processor products",
    "major semiconductor, display and other manufacturers",
    # geographic / regulatory
    "china", "taiwan", "germany", "singapore", "asia", "russia", "ukraine",
    "u.s.", "u.s. or foreign governments", "the united states",
    "the united states of america", "the european union", "the european economic area",
    "the u.s. department", "the u.s. department of commerce",
    "the u.s. department of commerce's", "the u.s. government",
    "the u.s. entity list", "the u.s. export administration regulations",
    "the foreign corrupt practices act", "the general data protection regulation",
    "the european commission", "the korean fair trade commission",
    "the california privacy protection agency",
    # product types / acronyms
    "gpu", "cpu", "ai", "cuda", "dram", "nand", "ssd", "hdd", "soc",
    "fpga", "asic", "rf", "gaas", "inp", "mems",
    # noise
    "2023", "3d", "•", "node",
}

# Regex for obviously junk entities
_JUNK_RE = re.compile(
    r"^(•|–|—|\d+|contents |the \w+ committee|the board|"
    r"title age|notes|f-\d+|segment technologies|"
    r"we are subject|existing repurchase|publicly announced|"
    r"internal control|financial statements|critical accounting)",
    re.IGNORECASE,
)


def filer_from_source(source_file: str) -> str:
    ticker = pathlib.Path(source_file).stem.split("_")[0].upper()
    return TICKER_TO_COMPANY.get(ticker, "")


def canonicalize(name: str) -> str:
    key = name.strip().lower().rstrip(".")
    return CANON.get(key, name.strip())


# Whitelist of legitimate companies we want in the supply chain graph.
# Only triples where AT LEAST ONE side is in this set are kept.
KNOWN_COMPANIES: set[str] = {
    # Our 8 filing companies
    "NVIDIA", "AMD", "Applied Materials", "Lam Research", "KLA",
    "Micron", "Broadcom", "Qualcomm", "Intel",
    # Key foundries
    "TSMC", "GLOBALFOUNDRIES", "UMC", "SMIC", "Samsung",
    # Equipment makers (non-filers)
    "ASML", "Tokyo Electron", "ASM International", "Hitachi",
    "Advantest America Inc.", "Lasertec", "Onto Innovation",
    "Screen Holdings", "Chroma ATE Inc.",
    # Assembly / packaging
    "Amkor Technology", "Tongfu Microelectronics", "Flex",
    # Key tech companies (customers / partners)
    "Microsoft", "Apple", "Dell", "HPE", "Sony", "IBM",
    "Hewlett Packard Enterprise", "HP",
    # Key acquisitions / subsidiaries
    "SoftBank", "Arm", "Xilinx", "VMware", "Pensando Systems",
    "Novellus", "LSI Logic Corporation",
    "Integrated Device Technology, Inc.",
    "Microsemi Corporation", "Ibiden Co",
    # Other notable supply chain players
    "Huawei", "BYD Auto",
    "Western Digital Corporation", "SanDisk Corporation",
    "Microchip", "Atlassian Corporation",
}

_PRONOUN_RE = re.compile(
    r"^(<filing company>|filing company|the company|the corporation|we|us|our company)$",
    re.IGNORECASE,
)


def resolve_entity(name: str, filer: str) -> str | None:
    """Return canonical entity name or None if it should be discarded."""
    name = name.strip()
    if not name:
        return None
    # Resolve pronouns to filer
    if _PRONOUN_RE.match(name):
        return filer if filer else None
    # Canonicalize
    canon = canonicalize(name)
    # Reject noise
    if canon.lower() in _NOISE:
        return None
    if _JUNK_RE.match(canon):
        return None
    # Reject very short strings
    if len(canon) <= 2:
        return None
    return canon


def clean_and_merge():
    rb_raw  = json.loads((ROOT / "rule_based_triples.json").read_text(encoding="utf-8"))["triples"]
    llm_raw = json.loads((ROOT / "groq_triples.json").read_text(encoding="utf-8"))["triples"]

    print(f"Raw input: {len(rb_raw)} rule-based + {len(llm_raw)} Groq = {len(rb_raw)+len(llm_raw)} total")

    merged: dict[tuple, dict] = {}  # (head, rel, tail) → best triple

    def process(triples, source_label):
        kept = skipped = 0
        for t in triples:
            filer = filer_from_source(t.get("source_file", ""))
            head  = resolve_entity(t["head"], filer)
            tail  = resolve_entity(t["tail"], filer)
            rel   = t["relation"]
            conf  = float(t.get("confidence", 0.5))

            if not head or not tail:
                skipped += 1; continue
            if head == tail:
                skipped += 1; continue  # self-loop
            if rel not in {
                "supplier_of","customer_of","subsidiary_of",
                "competitor_of","partner_of","depends_on"
            }:
                skipped += 1; continue

            # Both sides must be known companies for a clean graph
            if head not in KNOWN_COMPANIES or tail not in KNOWN_COMPANIES:
                skipped += 1; continue

            key = (head, rel, tail)
            if key not in merged or conf > merged[key]["confidence"]:
                merged[key] = {
                    "head": head, "relation": rel, "tail": tail,
                    "confidence": conf,
                    "source_file": t.get("source_file", ""),
                    "source_sentence": t.get("source_sentence", ""),
                    "extractor": source_label,
                }
                kept += 1
        return kept, skipped

    rb_kept,  rb_skip  = process(rb_raw,  "rule_based")
    llm_kept, llm_skip = process(llm_raw, "groq")

    triples = list(merged.values())
    triples.sort(key=lambda x: (-x["confidence"], x["head"], x["relation"]))

    out = ROOT / "merged_triples.json"
    out.write_text(
        json.dumps({"total": len(triples), "triples": triples}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nRule-based : {rb_kept} kept, {rb_skip} removed")
    print(f"Groq       : {llm_kept} kept, {llm_skip} removed")
    print(f"\nAfter merge & dedup: {len(triples)} unique triples")
    print(f"Saved → results/merged_triples.json")

    # Quick summary
    import collections
    rel_counts = collections.Counter(t["relation"] for t in triples)
    ents = {t["head"] for t in triples} | {t["tail"] for t in triples}
    print(f"\nRelation breakdown:")
    for rel, cnt in sorted(rel_counts.items(), key=lambda x: -x[1]):
        print(f"  {rel:<20} {cnt}")
    print(f"\nUnique companies in graph: {len(ents)}")
    print(f"  {sorted(ents)}")
    return triples


if __name__ == "__main__":
    clean_and_merge()

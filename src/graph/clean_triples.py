"""
Clean and merge rule-based + Groq triples into one high-quality dataset.

Strategy:
  - At least ONE side must be an anchor company (our 8 filers + key partners)
  - The OTHER side must pass a quality filter (looks like a real company name)
  - Canonicalize name variants (TSMC / Taiwan Semiconductor / etc.)
  - Remove self-loops, generic terms, product names, acronyms
  - Deduplicate keeping highest confidence
"""

import json, pathlib, re, collections

ROOT = pathlib.Path(__file__).parent.parent.parent / "results"

# ── Canonical name map ───────────────────────────────────────────────────────
CANON: dict[str, str] = {
    # TSMC
    "taiwan semiconductor manufacturing company limited": "TSMC",
    "taiwan semiconductor manufacturing company limited (tsmc)": "TSMC",
    "taiwan semiconductor manufacturing company": "TSMC",
    "taiwan semiconductor": "TSMC",
    "tsmc": "TSMC",
    # NVIDIA
    "nvidia corporation": "NVIDIA", "nvidia corp": "NVIDIA",
    "nvidia corp.": "NVIDIA", "nvidia": "NVIDIA", "nvda": "NVIDIA",
    # AMD
    "advanced micro devices, inc.": "AMD", "advanced micro devices, inc": "AMD",
    "advanced micro devices": "AMD", "amd": "AMD",
    # Applied Materials
    "applied materials, inc.": "Applied Materials",
    "applied materials, inc": "Applied Materials",
    "applied materials": "Applied Materials", "applied": "Applied Materials",
    # Lam Research
    "lam research corporation": "Lam Research", "lam research corp": "Lam Research",
    "lam research corp.": "Lam Research", "lam research": "Lam Research",
    "lam's": "Lam Research", "lam": "Lam Research",
    # KLA
    "kla corporation": "KLA", "kla corp": "KLA", "kla corp.": "KLA",
    "kla-tencor": "KLA", "kla's": "KLA", "kla": "KLA",
    # Micron
    "micron technology, inc.": "Micron", "micron technology, inc": "Micron",
    "micron technology": "Micron", "micron": "Micron",
    # Broadcom
    "broadcom corporation": "Broadcom", "broadcom corp.": "Broadcom",
    "broadcom corp": "Broadcom", "broadcom inc.": "Broadcom",
    "broadcom inc": "Broadcom", "broadcom": "Broadcom",
    # Qualcomm
    "qualcomm incorporated": "Qualcomm", "qualcomm inc.": "Qualcomm",
    "qualcomm inc": "Qualcomm", "qualcomm": "Qualcomm",
    # Intel
    "intel corporation": "Intel", "intel corp": "Intel",
    "intel corp.": "Intel", "intel": "Intel",
    # ASML
    "asml holding n.v.": "ASML", "asml holding": "ASML", "asml": "ASML",
    # Samsung
    "samsung electronics co., ltd.": "Samsung",
    "samsung electronics": "Samsung", "samsung": "Samsung",
    "samsung foundry": "Samsung",
    # Microsoft
    "microsoft corporation": "Microsoft", "microsoft": "Microsoft",
    # Apple
    "apple inc.": "Apple", "apple inc": "Apple", "apple": "Apple",
    # Sony
    "sony corporation": "Sony", "sony": "Sony",
    "sony interactive entertainment": "Sony",
    # SoftBank
    "softbank group corp": "SoftBank", "softbank group": "SoftBank",
    "softbank": "SoftBank",
    # Arm
    "arm limited": "Arm", "arm holdings": "Arm",
    "arm holdings plc": "Arm", "arm": "Arm",
    # GLOBALFOUNDRIES
    "globalfoundries inc.": "GLOBALFOUNDRIES",
    "globalfoundries inc. (gf)": "GLOBALFOUNDRIES",
    "globalfoundries": "GLOBALFOUNDRIES", "gf": "GLOBALFOUNDRIES",
    # UMC
    "united microelectronics corporation": "UMC", "umc": "UMC",
    # SMIC
    "semiconductor manufacturing international corporation": "SMIC",
    "smic": "SMIC",
    # VMware
    "vmware, inc.": "VMware", "vmware": "VMware",
    # Xilinx
    "xilinx, inc.": "Xilinx", "xilinx": "Xilinx",
    # Pensando
    "pensando systems, inc.": "Pensando Systems",
    "pensando systems": "Pensando Systems", "pensando": "Pensando Systems",
    # Dell
    "dell technologies": "Dell", "dell inc.": "Dell", "dell": "Dell",
    # HP / HPE
    "hewlett packard enterprise": "HPE", "hewlett-packard": "HP",
    "hewlett-packard corp": "HP", "hp inc.": "HP",
    # Google
    "google llc": "Google", "google": "Google", "alphabet": "Google",
    # Amazon
    "amazon web services": "AWS", "amazon": "Amazon", "aws": "AWS",
    # Meta
    "meta platforms": "Meta", "meta": "Meta", "facebook": "Meta",
    # Tokyo Electron
    "tokyo electron, ltd.": "Tokyo Electron",
    "tokyo electron limited": "Tokyo Electron", "tokyo electron": "Tokyo Electron",
    # Screen Holdings
    "screen holding co., ltd.": "Screen Holdings",
    "screen semiconductor solutions": "Screen Holdings",
    # Onto Innovation
    "onto innovation, inc.": "Onto Innovation",
    "onto innovation": "Onto Innovation",
    # Lasertec
    "lasertec, inc.": "Lasertec", "lasertec": "Lasertec",
    # Hitachi
    "hitachi high-technologies corporation": "Hitachi",
    "hitachi high-tech": "Hitachi", "hitachi": "Hitachi",
    # ASM International
    "asm international n.v.": "ASM International",
    "asm international": "ASM International", "asmi": "ASM International",
    # Flex
    "flex ltd.": "Flex", "flex limited": "Flex", "flex": "Flex",
    # GLOBALFOUNDRIES variants
    "globalfoundries inc.": "GLOBALFOUNDRIES",
    # Huawei variants
    "huawei technologies co., ltd.": "Huawei",
    # Xilinx variants
    "xilinx, inc.": "Xilinx",
    # Novellus variants
    "novellus systems, inc.": "Novellus Systems",
    # Onto Innovation variants
    "onto innovation, inc.": "Onto Innovation",
    # Lasertec variants
    "lasertec, inc.": "Lasertec",
    # KLA possessive
    "kla's": "KLA",
    # Lam possessive
    "lam's": "Lam Research",
    # ASML variant
    "asml holding n.v.": "ASML",
    # Advantest variant
    "advantest america inc.": "Advantest",
    # Amkor
    "amkor technology, inc.": "Amkor Technology",
    "amkor technology": "Amkor Technology", "amkor": "Amkor Technology",
    # Tongfu
    "tongfu microelectronics co., ltd": "Tongfu Microelectronics",
    "tongfu microelectronics": "Tongfu Microelectronics",
    # Advantest
    "advantest corporation": "Advantest", "advantest america inc.": "Advantest",
    "advantest": "Advantest",
    # Chroma
    "chroma ate inc.": "Chroma ATE", "chroma ate": "Chroma ATE",
    # Ibiden
    "ibiden co., ltd.": "Ibiden", "ibiden co": "Ibiden", "ibiden": "Ibiden",
    # Western Digital
    "western digital corporation": "Western Digital",
    "western digital": "Western Digital",
    # SanDisk
    "sandisk corporation": "SanDisk", "sandisk": "SanDisk",
    # Huawei
    "huawei technologies co., ltd.": "Huawei",
    "huawei technologies": "Huawei", "huawei": "Huawei",
    # Novellus
    "novellus systems, inc.": "Novellus Systems",
    "novellus systems": "Novellus Systems", "novellus": "Novellus Systems",
    # LSI Logic
    "lsi logic corporation": "LSI Logic", "lsi logic": "LSI Logic",
    # AT&T
    "at&t technologies": "AT&T", "at&t": "AT&T",
    # Microchip Technology
    "microchip technology incorporated": "Microchip Technology",
    "microchip technology": "Microchip Technology",
    "microchip": "Microchip Technology",
    # Microsemi
    "microsemi corporation": "Microsemi", "microsemi": "Microsemi",
    # IDT
    "integrated device technology, inc.": "IDT",
    "integrated device technology": "IDT",
    # Aruba Networks
    "aruba networks": "Aruba Networks",
    # CA Technologies
    "ca technologies": "CA Technologies",
    # Xilinx Notes
    "the assumed xilinx notes": None,   # document reference, not a company
    # Xbox / PlayStation (products → parent companies)
    "xbox": "Microsoft",
    "playstation": "Sony",
    # BYD
    "byd auto": "BYD Auto", "byd": "BYD Auto",
    # KYEC
    "kyec": "KYEC",
    # Atlassian
    "atlassian corporation plc": "Atlassian",
    "atlassian corporation": "Atlassian", "atlassian": "Atlassian",
}

# Filer ticker → canonical company name
TICKER_TO_COMPANY: dict[str, str] = {
    "LRCX": "Lam Research", "AMAT": "Applied Materials", "KLAC": "KLA",
    "MU": "Micron", "NVDA": "NVIDIA", "AMD": "AMD",
    "AVGO": "Broadcom", "QCOM": "Qualcomm", "INTC": "Intel",
}

# Anchor companies — at least one side of a triple must be in this set
ANCHOR_COMPANIES: set[str] = {
    # Our 8 filing companies
    "NVIDIA", "AMD", "Applied Materials", "Lam Research", "KLA",
    "Micron", "Broadcom", "Qualcomm", "Intel",
    # Key foundries & chip makers
    "TSMC", "GLOBALFOUNDRIES", "UMC", "SMIC", "Samsung",
    # Key equipment makers
    "ASML", "Tokyo Electron", "ASM International", "Hitachi",
    "Advantest", "Lasertec", "Onto Innovation", "Screen Holdings", "Chroma ATE",
    # Assembly / packaging
    "Amkor Technology", "Tongfu Microelectronics", "Flex", "Ibiden",
    # Major tech customers / partners
    "Microsoft", "Apple", "Dell", "HPE", "Sony", "IBM", "Google", "AWS",
    "Amazon", "Meta",
    # Key subsidiaries / acquisitions
    "SoftBank", "Arm", "Xilinx", "VMware", "Pensando Systems",
    "Novellus Systems", "LSI Logic", "Microsemi", "IDT",
    "Atlassian", "CA Technologies", "Aruba Networks",
    # Storage
    "Western Digital", "SanDisk",
    # Other meaningful supply chain companies
    "Huawei", "BYD Auto", "Microchip Technology", "AT&T",
    "KYEC",
}

# Generic / non-company noise words — always rejected
_NOISE: set[str] = {
    "we", "us", "our", "the company", "company", "the corporation",
    "customers", "customer", "suppliers", "supplier", "vendors", "vendor",
    "foundries", "foundry", "partners", "partner", "competitors", "competitor",
    "third parties", "third party", "third-party", "a third party",
    "manufacturers", "manufacturer", "subcontractors",
    "original device manufacturers", "odms", "oems", "oem",
    "industry leaders", "start-ups", "governmental entities",
    "employees", "consultants", "retailers/distributors", "system builders",
    "semiconductor manufacturers", "our customers", "our suppliers",
    "outsource providers", "other service providers",
    "tier-1 automotive suppliers", "automotive manufacturers",
    "foundry/logic customers",
    # geographic / regulatory
    "china", "taiwan", "germany", "singapore", "asia", "russia", "ukraine",
    "u.s.", "u.s. or foreign governments", "the united states",
    "the united states of america", "the european union",
    "the u.s. department", "the u.s. department of commerce",
    "the u.s. government", "the european commission",
    # pure product types
    "gpu", "cpu", "ai", "cuda", "dram", "nand", "ssd", "hdd", "soc",
    "fpga", "asic", "rf", "gaas", "inp", "mems", "euv", "hbm",
    "geforce", "ryzen", "radeon", "hopper", "alveo",
    # concepts / misc
    "2023", "3d", "•", "node", "covid-19", "software", "hardware",
    "moore's law", "pc supply chain", "data center", "labor", "assembly",
    "california", "lehi", "fort collins", "hong kong", "xinjiang",
    "board of directors", "board meetings", "sec", "esg",
    # business/finance terms that aren't companies
    "customer opportunity", "dispose of businesses", "business combinations",
    "stock purchase plan", "repurchase program", "credit facility",
    "equity incentive plan", "long-term incentive plan",
}

# Junk patterns — always rejected
_JUNK_RE = re.compile(
    r"^(•|–|—|\d[\d\s]*$|contents\s|the \w+ committee|the board\b|"
    r"title age|notes\b|f-\d|segment technologies|"
    r"we are subject|existing repurchase|publicly announced|"
    r"internal control|financial statements|critical accounting|"
    r"ir\.|[a-z]{1,3}\.[a-z]{2,}|"           # websites like ir.kla.com
    r"scope \d|ras?u|ghg|pfas|fbar|sg&a|"    # accounting/ESG acronyms
    r"rd&e|rpo|nvm|wlp|ics|igp|stb|bios|"
    r"the \w+ \w+ \w+ act|"                   # legal acts
    r"the \w+ \w+ \w+ regulation)",
    re.IGNORECASE,
)

# Short all-caps strings that are NOT known company abbreviations → likely noise
_CAPS_NOISE_RE = re.compile(r"^[A-Z0-9]{1,4}$")
_KNOWN_CAPS = {
    "AMD", "KLA", "IBM", "HPE", "UMC", "AWS", "IDT", "HP", "AT&T",
    "TSMC", "ASML", "SMIC", "KYEC", "BYD",
}


def filer_from_source(source_file: str) -> str:
    ticker = pathlib.Path(source_file).stem.split("_")[0].upper()
    return TICKER_TO_COMPANY.get(ticker, "")


# Pre-normalise CANON keys so lookups are consistent regardless of trailing punctuation
_CANON_INDEX: dict[str, str | None] = {
    k.strip().lower().rstrip(".,; "): v for k, v in CANON.items()
}


def canonicalize(name: str) -> str | None:
    key = name.strip().lower().rstrip(".,; ")
    # Also try stripping possessive apostrophe-s
    key_no_poss = re.sub(r"'s$", "", key)
    result = _CANON_INDEX.get(key) or _CANON_INDEX.get(key_no_poss)
    if result is None and key not in _CANON_INDEX and key_no_poss not in _CANON_INDEX:
        return name.strip()   # not in map → return as-is
    return result  # None means explicitly blacklisted


_PRONOUN_RE = re.compile(
    r"^(<filing company>|filing company|the company|the corporation|"
    r"we|us|our company|the registrant)$",
    re.IGNORECASE,
)


_GENERIC_PHRASES = re.compile(
    r"^(third.party|our |another |open source|public cloud|smaller companies|"
    r"diversified|critical infra|internet and|mapping compan|"
    r"providers of|companies that|major semiconductor|customers in|"
    r"foundry.logic|more compan|other compan|new compan)",
    re.IGNORECASE,
)

# Product-name patterns — AMD Ryzen X, NVIDIA Omniverse, etc.
_PRODUCT_RE = re.compile(
    r"^(AMD |NVIDIA |Intel |Qualcomm |Broadcom |Lam |KLA |Applied ).+",
    re.IGNORECASE,
)
_PRODUCT_EXCEPTIONS = {
    "AMD", "NVIDIA", "Intel", "Qualcomm", "Broadcom",
    "Lam Research", "KLA", "Applied Materials",
}

# Categories / plural acronym sets that aren't individual companies
_CATEGORY_SUFFIXES = {"AIBs", "CMs", "IDMs", "ISVs", "OEMs", "ODMs", "JVs",
                       "CSPs", "SoCs", "GPUs", "CPUs", "DPUs"}


def is_valid_entity(name: str) -> bool:
    """Quality gate for non-anchor entities — must look like a real company name."""
    if len(name) < 3:
        return False
    # Short all-caps strings not in known list are usually acronyms/products
    if _CAPS_NOISE_RE.match(name) and name not in _KNOWN_CAPS:
        return False
    # Must contain at least one letter
    if not re.search(r"[A-Za-z]", name):
        return False
    # Reject strings starting with "the" that aren't proper company names
    if name.lower().startswith("the ") and name not in ANCHOR_COMPANIES:
        return False
    # Reject bullet/symbol starts
    if name[0] in "•–—-/":
        return False
    # Reject generic multi-word descriptions
    if _GENERIC_PHRASES.match(name):
        return False
    # Reject product names (e.g. "AMD Ryzen 5000 Series", "NVIDIA Omniverse")
    if _PRODUCT_RE.match(name) and name not in _PRODUCT_EXCEPTIONS:
        return False
    # Reject category plurals
    if name in _CATEGORY_SUFFIXES or any(name.endswith(s) for s in _CATEGORY_SUFFIXES):
        return False
    # Reject things that are clearly not company names (universities, standards, events)
    _NOT_COMPANIES = {
        "Stanford University", "Moore's Law", "Moody's Investors Service",
        "Responsible Business Alliance", "Revolving Credit Facility",
        "VMware Merger", "Ethereum", "Wi-Fi", "H.264",
        "ISO 45001", "FactSet financial data", "Seshasayee",
        "Field Programmable Gate Arrays", "Data Over Cable Service Interface Specifications",
        "Identity & Access Management", "Employee Health and Safety",
        "Employee Resource Groups", "Employees' Stock Purchase Plan",
        "Integration Risks Acquisitions", "Licenses Protection",
        "Decline", "Augmented", "DevOps", "FC SAN", "Coreware",
        "GPU Cloud", "GeForce NOW", "GNSS semiconductor",
        "NVIDIA AI Enterprise", "NVIDIA Omniverse", "NVIDIA's partner network",
        "Customer Semiconductor", "Consolidated Financial Statements",
        "Ultimate", "WRX80", "ASC 805", "Xi'an", "ATMP JVs",
    }
    if name in _NOT_COMPANIES:
        return False
    # Reject long generic descriptions (>6 words = probably a category description)
    if len(name.split()) > 6:
        return False
    return True


def _norm_apostrophe(s: str) -> str:
    """Replace Unicode smart-quote apostrophes with ASCII so dict lookups work."""
    return s.replace("’", "'").replace("‘", "'").replace("ʼ", "'")


def resolve_entity(name: str, filer: str) -> str | None:
    name = _norm_apostrophe(name.strip())
    if not name:
        return None
    # Resolve filing-company pronouns
    if _PRONOUN_RE.match(name):
        return filer if filer else None
    # Canonicalize (may return None if explicitly blacklisted)
    canon = canonicalize(name)
    if canon is None:
        return None
    # Noise list
    if canon.lower() in _NOISE:
        return None
    # Junk regex
    if _JUNK_RE.match(canon):
        return None
    return canon


# ── Main merge function ──────────────────────────────────────────────────────

def clean_and_merge() -> list[dict]:
    rb_raw  = json.loads((ROOT / "rule_based_triples.json").read_text(encoding="utf-8"))["triples"]
    llm_raw = json.loads((ROOT / "groq_triples.json").read_text(encoding="utf-8"))["triples"]

    print(f"Raw input: {len(rb_raw)} rule-based + {len(llm_raw)} Groq = {len(rb_raw)+len(llm_raw)} total")

    VALID_RELS = {
        "supplier_of","customer_of","subsidiary_of",
        "competitor_of","partner_of","depends_on",
    }

    merged: dict[tuple, dict] = {}

    stats = collections.defaultdict(int)

    def process(triples, label):
        kept = 0
        for t in triples:
            filer = filer_from_source(t.get("source_file",""))
            head  = resolve_entity(t["head"], filer)
            tail  = resolve_entity(t["tail"], filer)
            rel   = t["relation"]
            conf  = float(t.get("confidence", 0.5))

            if not head or not tail:
                stats[f"{label}_no_entity"] += 1; continue
            if head == tail:
                stats[f"{label}_self_loop"] += 1; continue
            if rel not in VALID_RELS:
                stats[f"{label}_bad_rel"] += 1; continue

            # Anchor rule: at least one side must be a known anchor company
            head_anchored = head in ANCHOR_COMPANIES
            tail_anchored = tail in ANCHOR_COMPANIES
            if not head_anchored and not tail_anchored:
                stats[f"{label}_no_anchor"] += 1; continue

            # Quality gate for the non-anchor side
            if not head_anchored and not is_valid_entity(head):
                stats[f"{label}_bad_entity"] += 1; continue
            if not tail_anchored and not is_valid_entity(tail):
                stats[f"{label}_bad_entity"] += 1; continue

            key = (head, rel, tail)
            if key not in merged or conf > merged[key]["confidence"]:
                merged[key] = {
                    "head": head, "relation": rel, "tail": tail,
                    "confidence": conf,
                    "source_file": t.get("source_file",""),
                    "source_sentence": t.get("source_sentence",""),
                    "extractor": label,
                }
                kept += 1
        return kept

    rb_kept  = process(rb_raw,  "rule_based")
    llm_kept = process(llm_raw, "groq")

    triples = list(merged.values())
    triples.sort(key=lambda x: (-x["confidence"], x["head"], x["relation"]))

    out = ROOT / "merged_triples.json"
    out.write_text(
        json.dumps({"total": len(triples), "triples": triples}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Summary
    rel_counts = collections.Counter(t["relation"] for t in triples)
    ents = sorted({t["head"] for t in triples} | {t["tail"] for t in triples})

    print(f"\n  Rule-based : {rb_kept} kept  |  filter breakdown: {dict((k,v) for k,v in stats.items() if 'rule' in k)}")
    print(f"  Groq       : {llm_kept} kept  |  filter breakdown: {dict((k,v) for k,v in stats.items() if 'groq' in k)}")
    print(f"\n  After merge & dedup  : {len(triples)} unique clean triples")
    print(f"  Unique companies     : {len(ents)}")
    print(f"\n  Relation breakdown:")
    for rel, cnt in sorted(rel_counts.items(), key=lambda x: -x[1]):
        bar = "█" * cnt
        print(f"    {rel:<20} {cnt:3d}  {bar}")
    print(f"\n  All companies in graph:")
    for e in ents:
        anchor = " ★" if e in ANCHOR_COMPANIES else ""
        print(f"    {e}{anchor}")

    print(f"\n  Saved → {out}")
    return triples


if __name__ == "__main__":
    clean_and_merge()

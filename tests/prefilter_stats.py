"""Print the sentence pre-filter stats for all 8 filings (no API calls)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
try:
    from dotenv import load_dotenv; load_dotenv(pathlib.Path(__file__).parent.parent / ".env")
except ImportError:
    pass

from baselines.zero_shot_llm import sentence_filter_stats

MAX_SEND = 80  # matches extract_file default

FILINGS = sorted(pathlib.Path("data/filings").glob("*_2023.txt"))

print(f"\n{'Filing':<22}  {'Total':>6}  {'Relevant':>9}  {'% relevant':>11}  {'Sending':>8}")
print("-" * 68)
grand_total = grand_rel = grand_send = 0
for fp in FILINGS:
    total, n_rel, rel_sents = sentence_filter_stats(fp)
    send = min(n_rel, MAX_SEND)
    pct = 100 * n_rel / total if total else 0.0
    cap = " (capped)" if n_rel > MAX_SEND else ""
    print(f"  {fp.name:<20}  {total:6d}  {n_rel:9d}  {pct:10.1f}%  {send:8d}{cap}")
    grand_total += total; grand_rel += n_rel; grand_send += send

print("-" * 68)
g_pct = 100 * grand_rel / grand_total if grand_total else 0.0
print(f"  {'TOTAL':<20}  {grand_total:6d}  {grand_rel:9d}  {g_pct:10.1f}%  {grand_send:8d}")
print(f"\n  At ~0.3 s/call that is ~{grand_send * 0.3:.0f} s of API time for the full run.\n")

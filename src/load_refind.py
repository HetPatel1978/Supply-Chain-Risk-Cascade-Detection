"""
Load the REFinD dataset and save 80/10/10 train/val/test splits to data/refind/.

REFinD (Relation Extraction Financial Dataset, SIGIR 2023) is NOT on HuggingFace Hub.
Official page : https://refind-re.github.io/
Paper         : https://arxiv.org/abs/2305.18322

Download steps (one-time manual step):
  1. Go to https://refind-re.github.io/ and follow the CodaLab data link.
  2. Download and unzip into data/refind/raw/ so you have:
       data/refind/raw/train.json   (or train.csv / train.tsv)
       data/refind/raw/test.json
       (validation may or may not be included)
  3. Re-run this script — it will detect and load the local files automatically.

The script also tries known HuggingFace Hub IDs first (in case a mirror appears later).
"""

import json
import pathlib
from datasets import load_dataset, Dataset, DatasetDict

RAW_DIR = pathlib.Path(__file__).parent.parent / "data" / "refind" / "raw"
OUT_DIR = pathlib.Path(__file__).parent.parent / "data" / "refind"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HF_IDS = [
    "relbert/refind",
    "DFKI-SLT/REFinD",
    "nbroad/refind",
    "simerjotwrites/refind",
]


# ── loaders ──────────────────────────────────────────────────────────────────

def try_hf_load() -> DatasetDict | None:
    for hf_id in HF_IDS:
        try:
            print(f"  Trying HuggingFace: {hf_id}")
            ds = load_dataset(hf_id)
            print(f"  Loaded from {hf_id}")
            return ds
        except Exception:
            pass
    return None


def try_local_load() -> DatasetDict | None:
    """Look for JSON/CSV/TSV files in data/refind/raw/."""
    if not RAW_DIR.exists():
        return None
    json_files = list(RAW_DIR.glob("*.json")) + list(RAW_DIR.glob("*.jsonl"))
    csv_files  = list(RAW_DIR.glob("*.csv"))
    tsv_files  = list(RAW_DIR.glob("*.tsv"))
    if not (json_files or csv_files or tsv_files):
        return None

    print(f"  Found local files in {RAW_DIR}")
    splits = {}
    for split_name in ("train", "validation", "dev", "test"):
        for ext, fmt in ((".json", "json"), (".jsonl", "json"), (".csv", "csv"), (".tsv", "csv")):
            candidate = RAW_DIR / f"{split_name}{ext}"
            if candidate.exists():
                kwargs = {"data_files": str(candidate)}
                if fmt == "csv" and ext == ".tsv":
                    kwargs["delimiter"] = "\t"
                ds_split = load_dataset(fmt, **kwargs, split="train")
                splits[split_name if split_name != "dev" else "validation"] = ds_split
                print(f"    loaded {split_name}{ext} → {len(ds_split):,} rows")
                break
    # Also check for a single combined file
    if not splits:
        for f, fmt in [(json_files[0], "json")] if json_files else [(csv_files[0], "csv")] if csv_files else [(tsv_files[0], "csv")]:
            print(f"    Using single file: {f.name}")
            kwargs = {"data_files": str(f)}
            if fmt == "csv" and f.suffix == ".tsv":
                kwargs["delimiter"] = "\t"
            single = load_dataset(fmt, **kwargs, split="train")
            splits["all"] = single
            break

    return DatasetDict(splits) if splits else None


def ensure_splits(ds: DatasetDict) -> DatasetDict:
    if {"train", "validation", "test"}.issubset(ds.keys()):
        print("Pre-built train/validation/test splits found.")
        return ds

    # Merge everything into one pool then re-split
    all_data = None
    for v in ds.values():
        all_data = v if all_data is None else Dataset.from_dict(
            {col: all_data[col] + v[col] for col in all_data.column_names}
        )

    print(f"Performing 80/10/10 split on {len(all_data):,} total rows...")
    tmp      = all_data.train_test_split(test_size=0.20, seed=42)
    val_test = tmp["test"].train_test_split(test_size=0.50, seed=42)
    return DatasetDict(train=tmp["train"], validation=val_test["train"], test=val_test["test"])


def save_splits(ds: DatasetDict):
    for name, split in ds.items():
        path = OUT_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in split:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  Saved {name}.jsonl  ({len(split):,} rows) → {path}")


def show_samples(ds: DatasetDict, n: int = 3):
    print(f"\n--- {n} sample rows from train split ---")
    for i, row in enumerate(ds["train"].select(range(min(n, len(ds["train"]))))):
        print(f"\n[{i}]\n{json.dumps(row, indent=2, ensure_ascii=False)}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Attempting HuggingFace Hub load...")
    ds_raw = try_hf_load()

    if ds_raw is None:
        print("HuggingFace Hub load failed. Checking for local files...")
        ds_raw = try_local_load()

    if ds_raw is None:
        print(
            "\nREFinD dataset not found.\n"
            "Manual download required:\n"
            "  1. Visit https://refind-re.github.io/ and follow the CodaLab link.\n"
            "  2. Download and unzip into: data/refind/raw/\n"
            "  3. Re-run this script.\n"
        )
        return

    print(f"\nRaw dataset keys : {list(ds_raw.keys())}")
    print(f"Columns          : {ds_raw.column_names}")

    ds = ensure_splits(ds_raw)
    counts = {k: len(v) for k, v in ds.items()}
    print(f"\nSplit counts: {counts}")
    save_splits(ds)
    show_samples(ds)
    print("\nDone. Splits saved to data/refind/")


if __name__ == "__main__":
    main()

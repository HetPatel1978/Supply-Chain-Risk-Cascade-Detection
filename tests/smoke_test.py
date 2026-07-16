"""One-line smoke tests for each core library."""

print("--- transformers ---")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("roberta-base")
assert tok.encode("hello") is not None
print("OK: AutoTokenizer loaded roberta-base")

print("--- datasets ---")
from datasets import Dataset
ds = Dataset.from_dict({"text": ["supply chain", "risk cascade"], "label": [0, 1]})
assert len(ds) == 2
print(f"OK: Dataset created with {len(ds)} rows")

print("--- networkx ---")
import networkx as nx
G = nx.DiGraph()
G.add_edge("CompanyA", "CompanyB", relation="supplier_of")
assert nx.has_path(G, "CompanyA", "CompanyB")
print(f"OK: DiGraph with {G.number_of_nodes()} nodes, shortest path verified")

print("--- groq ---")
from groq import Groq
client = Groq.__new__(Groq)  # instantiate without API key to check import only
print("OK: groq.Groq class importable (API call skipped — set GROQ_API_KEY in .env)")

print("\nAll smoke tests passed.")

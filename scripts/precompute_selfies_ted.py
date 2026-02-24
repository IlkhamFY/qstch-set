"""Precompute SELFIES-TED embeddings for BTZ multi-redox dataset.

Saves embeddings as a .pt file for use in molecular_benchmark.py
without needing the transformers model at runtime.
"""
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import importlib.metadata
_orig_version = importlib.metadata.version
def _patched_version(name):
    if name == "numpy":
        import numpy
        return numpy.__version__
    return _orig_version(name)
importlib.metadata.version = _patched_version

import csv
from pathlib import Path
import torch
import selfies as sf
from transformers import AutoTokenizer, AutoModel

DATA_DIR = Path(__file__).parent.parent / "data"
CSV_PATH = DATA_DIR / "redox_mer.csv"
OUT_PATH = DATA_DIR / "redox_mer_selfies_ted.pt"

# Load SMILES
with open(CSV_PATH, "r") as f:
    rows = list(csv.DictReader(f))
smiles_list = [r["SMILES"] for r in rows]
print(f"Loaded {len(smiles_list)} molecules")

# Load model
print("Loading SELFIES-TED model...")
tokenizer = AutoTokenizer.from_pretrained("ibm/materials.selfies-ted")
model = AutoModel.from_pretrained("ibm/materials.selfies-ted")
model.eval()

# Encode all molecules
embeddings = []
failed = []
for i, smi in enumerate(smiles_list):
    try:
        selfies_str = sf.encoder(smi)
        if selfies_str is None:
            raise ValueError(f"SELFIES encoding failed for {smi}")
        selfies_str = selfies_str.replace("][", "] [")

        tok = tokenizer(
            selfies_str,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding="max_length",
        )

        with torch.no_grad():
            out = model.encoder(
                input_ids=tok["input_ids"],
                attention_mask=tok["attention_mask"],
            )
            hidden = out.last_hidden_state
            mask = tok["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        embeddings.append(pooled.squeeze(0))
    except Exception as e:
        print(f"  WARN: molecule {i} ({smi}): {e}")
        # Use zero vector as fallback
        embeddings.append(torch.zeros(1024))
        failed.append(i)

    if (i + 1) % 100 == 0:
        print(f"  Encoded {i+1}/{len(smiles_list)}")

X = torch.stack(embeddings)
print(f"\nEmbedding tensor: {X.shape}")
print(f"Failed molecules: {len(failed)}")
if failed:
    print(f"  Indices: {failed[:20]}...")

# Save
torch.save(X, OUT_PATH)
print(f"Saved to {OUT_PATH}")
print(f"File size: {OUT_PATH.stat().st_size / 1024 / 1024:.1f} MB")

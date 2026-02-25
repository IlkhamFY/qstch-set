"""Test SELFIES-TED embedding extraction."""
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Workaround for numpy version detection bug in transformers 5.2.0
import importlib.metadata
_orig_version = importlib.metadata.version
def _patched_version(name):
    if name == "numpy":
        import numpy
        return numpy.__version__
    return _orig_version(name)
importlib.metadata.version = _patched_version

from transformers import AutoTokenizer, AutoModel
import selfies as sf
import torch

print("Loading SELFIES-TED model...")
tokenizer = AutoTokenizer.from_pretrained("ibm/materials.selfies-ted")
model = AutoModel.from_pretrained("ibm/materials.selfies-ted")
model.eval()

# Test molecule: benzene
smiles = "c1ccccc1"
selfies_str = sf.encoder(smiles)
selfies_str = selfies_str.replace("][", "] [")
print(f"SMILES: {smiles}")
print(f"SELFIES: {selfies_str}")

# Tokenize
tokens = tokenizer(
    selfies_str,
    return_tensors="pt",
    max_length=128,
    truncation=True,
    padding="max_length",
)

# Encode
with torch.no_grad():
    outputs = model.encoder(
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
    )
    hidden = outputs.last_hidden_state  # (1, seq_len, embed_dim)

    # Mean pooling over non-padding tokens
    mask = tokens["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

print(f"Embedding dim: {pooled.shape[1]}")
print(f"Embedding shape: {pooled.shape}")
print(f"First 5 values: {pooled[0, :5].tolist()}")
print(f"Norm: {pooled.norm().item():.4f}")

# Test batch
smiles_list = ["c1ccccc1", "CCO", "CC(=O)O", "c1ccc(O)cc1"]
embeddings = []
for smi in smiles_list:
    sel = sf.encoder(smi).replace("][", "] [")
    tok = tokenizer(sel, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
    with torch.no_grad():
        out = model.encoder(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
        h = out.last_hidden_state
        m = tok["attention_mask"].unsqueeze(-1).expand(h.size()).float()
        emb = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
    embeddings.append(emb.squeeze(0))

X = torch.stack(embeddings)
print(f"\nBatch embeddings: {X.shape}")
print(f"All distinct: {len(set(tuple(e.tolist()) for e in X)) == len(smiles_list)}")
print("SELFIES-TED test passed!")

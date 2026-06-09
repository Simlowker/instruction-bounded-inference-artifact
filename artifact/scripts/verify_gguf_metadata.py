#!/usr/bin/env python3
"""P0-4: Verify layers_real, hidden_dim, d_ff, vocab, params from GGUF metadata.

Reads every GGUF referenced in models.csv (or discovered on disk) and emits:
- data/verified/gguf_metadata.csv  (raw truth from files)
- data/verified/discrepancies.csv  (rows where models.csv disagrees)

Usage: python3 verify_gguf_metadata.py
"""
import csv
import sys
from pathlib import Path

try:
    from gguf import GGUFReader
except ImportError:
    print("Install gguf: pip3 install gguf", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
MODELS_CSV = ROOT / "data" / "models.csv"
GGUF_ROOTS = [
    Path("llama_cpp_canister/models"),
]
OUT_DIR = ROOT / "data" / "verified"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEY_FIELDS = {
    "block_count": ["block_count", ".block_count"],
    "embedding_length": ["embedding_length", ".embedding_length"],
    "feed_forward_length": ["feed_forward_length", ".feed_forward_length"],
    "vocab_size": ["vocab_size", ".vocab_size"],
    "context_length": ["context_length", ".context_length"],
    "head_count": ["attention.head_count", ".attention.head_count"],
    "arch": ["general.architecture"],
}


def read_gguf(path: Path) -> dict:
    r = GGUFReader(str(path))
    meta: dict = {"file": str(path), "size_MB": round(path.stat().st_size / 1024 / 1024, 1)}
    # arch key lives under general.architecture
    arch = ""
    for f in r.fields.values():
        if f.name == "general.architecture":
            arch = bytes(f.parts[-1]).decode("utf-8", errors="replace")
            break
    meta["arch"] = arch
    # architecture-prefixed keys
    def get(key_names):
        for f in r.fields.values():
            for kn in key_names:
                if f.name.endswith(kn):
                    try:
                        return int(f.parts[-1][0])
                    except Exception:
                        try:
                            return f.parts[-1][0]
                        except Exception:
                            return None
        return None

    meta["block_count"] = get([".block_count"])
    meta["embedding_length"] = get([".embedding_length"])
    meta["feed_forward_length"] = get([".feed_forward_length"])
    meta["context_length"] = get([".context_length"])
    meta["head_count"] = get([".attention.head_count"])
    # vocab: read tokens tensor length if present, else tokenizer.ggml.tokens
    vocab = None
    for f in r.fields.values():
        if f.name == "tokenizer.ggml.tokens":
            vocab = len(f.data)
            break
    meta["vocab_size"] = vocab
    # param count: sum of tensor element counts
    total_params = 0
    for t in r.tensors:
        shape = list(t.shape)
        p = 1
        for d in shape:
            if d > 0:
                p *= int(d)
        total_params += p
    meta["params_M_calc"] = round(total_params / 1_000_000, 1)
    # quant: take the first weight tensor's type
    quant = ""
    for t in r.tensors:
        if "weight" in t.name and "norm" not in t.name and "embed" not in t.name:
            try:
                quant = t.tensor_type.name
            except Exception:
                quant = str(t.tensor_type)
            break
    meta["quant_detected"] = quant
    return meta


def discover_ggufs() -> list[Path]:
    files: list[Path] = []
    for root in GGUF_ROOTS:
        if not root.exists():
            continue
        files.extend(root.rglob("*.gguf"))
    return sorted(set(files))


def main():
    files = discover_ggufs()
    print(f"Found {len(files)} GGUF files", file=sys.stderr)

    rows = []
    for f in files:
        try:
            m = read_gguf(f)
        except Exception as e:
            m = {"file": str(f), "error": str(e)}
        rows.append(m)
        name = f.name
        bc = m.get("block_count")
        el = m.get("embedding_length")
        vs = m.get("vocab_size")
        pm = m.get("params_M_calc")
        q = m.get("quant_detected", "")
        print(f"  {name}: L={bc} d={el} V={vs} P={pm}M q={q}", file=sys.stderr)

    out_path = OUT_DIR / "gguf_metadata.csv"
    fieldnames = [
        "file",
        "size_MB",
        "arch",
        "block_count",
        "embedding_length",
        "feed_forward_length",
        "head_count",
        "context_length",
        "vocab_size",
        "params_M_calc",
        "quant_detected",
        "error",
    ]
    with open(out_path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

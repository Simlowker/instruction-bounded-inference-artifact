#!/usr/bin/env python3
"""
MTEB(fra, v1) off-chain baseline for EmbeddingGemma-300M.

Runs Retrieval, Reranking, STS tasks from MTEB(fra, v1) benchmark.
Saves per-task scores to CSV + JSON in artifact/mteb/.

Usage:
    # From artifact/ directory, with mteb-venv activated:
    python scripts/run_mteb_fra.py

    # Custom model:
    python scripts/run_mteb_fra.py --model intfloat/multilingual-e5-small

    # Specific task types:
    python scripts/run_mteb_fra.py --task-types Retrieval STS

Requirements:
    pip install mteb sentence-transformers torch pandas

Auth (for gated models):
    huggingface-cli login
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TASK_TYPES_DEFAULT = ["Retrieval", "Reranking", "STS"]
OUTDIR_DEFAULT = Path(__file__).resolve().parent.parent / "mteb"


def get_versions() -> dict:
    import mteb
    import sentence_transformers
    import torch
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "mteb": mteb.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run(model_name: str, outdir: Path, task_types: list[str]):
    import mteb
    import pandas as pd

    outdir.mkdir(parents=True, exist_ok=True)
    versions = get_versions()

    # Load benchmark and filter tasks
    benchmark = mteb.get_benchmark("MTEB(fra, v1)")
    tasks = [t for t in benchmark.tasks if t.metadata.type in task_types]

    print(f"Model: {model_name}")
    print(f"Tasks: {len(tasks)} ({', '.join(task_types)})")
    for t in tasks:
        print(f"  {t.metadata.type:15s} | {t.metadata.name}")
    print()

    # Load model
    print(f"Loading model...")
    model = mteb.get_model(model_name)

    # Run evaluation task by task (saves intermediate results)
    all_scores = []
    t0 = time.time()

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Running: {task.metadata.name} ({task.metadata.type})")
        print(f"{'='*60}")

        try:
            result = mteb.evaluate(
                model=model,
                tasks=task,
                raise_error=True,
                show_progress_bar=True,
            )

            # Extract scores from result
            for split, scores in result.scores.items():
                for score_dict in scores:
                    row = {
                        "task": result.task_name,
                        "type": task.metadata.type,
                        "split": split,
                        "main_score": score_dict.get("main_score"),
                    }
                    # Add individual metric scores
                    for k, v in score_dict.items():
                        if k not in ("main_score", "hf_subset", "languages") and isinstance(v, (int, float)):
                            row[k] = v
                    all_scores.append(row)

            print(f"  -> main_score: {all_scores[-1]['main_score']:.4f}")
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            all_scores.append({
                "task": task.metadata.name,
                "type": task.metadata.type,
                "split": "error",
                "main_score": None,
                "error": str(e),
            })

    elapsed = time.time() - t0

    # Save CSV
    df = pd.DataFrame(all_scores)
    csv_path = outdir / "mteb_fra_scores.csv"
    df.to_csv(csv_path, index=False)

    # Save JSON with metadata
    output = {
        "model": model_name,
        "benchmark": "MTEB(fra, v1)",
        "task_types": task_types,
        "versions": versions,
        "elapsed_seconds": round(elapsed, 1),
        "scores": all_scores,
    }
    json_path = outdir / "mteb_fra_scores.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(df[["task", "type", "split", "main_score"]].to_string(index=False))
    print(f"\nElapsed: {elapsed:.0f}s")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")

    return output


def main():
    parser = argparse.ArgumentParser(description="MTEB(fra) off-chain baseline")
    parser.add_argument("--model", default="google/EmbeddingGemma-300M")
    parser.add_argument("--outdir", type=Path, default=OUTDIR_DEFAULT)
    parser.add_argument("--task-types", nargs="+", default=TASK_TYPES_DEFAULT)
    args = parser.parse_args()

    run(args.model, args.outdir, args.task_types)


if __name__ == "__main__":
    main()

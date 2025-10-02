from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

EVAL_DIR = Path("evals")

PROP = "avg_acc,none"

@dataclass
class ModelEval:
    tag: str
    method: str
    k: int | None
    score: float
    source: Path

    @property
    def label(self) -> str:
        if self.method == "base":
            return "Base"
        if self.method == "vanilla":
            return f"Vanilla k={self.k}" if self.k is not None else "Vanilla"
        if self.method == "top-k-tok":
            return f"Top-k {self.k}%" if self.k is not None else "Top-k"
        return self.tag


def infer_method(tag: str) -> tuple[str, int | None]:
    tag_lower = tag.lower()
    # Base model (non-distilled)
    if tag_lower in {"qwen3-0.6b", "base"}:
        return "base", None

    # Vanilla distillation (all tokens)
    vanilla_match = re.search(r"vanilla(?:_k(\d+))?", tag_lower)
    if vanilla_match:
        k_val = vanilla_match.group(1)
        return "vanilla", int(k_val) if k_val else None

    # Top-k entropy token selection
    topk_match = re.search(r"top_k_tok_k(\d+)", tag_lower)
    if topk_match:
        return "top-k-tok", int(topk_match.group(1))

    return "other", None


def load_model_evals() -> List[ModelEval]:
    if not EVAL_DIR.exists():
        raise FileNotFoundError(f"Eval directory not found: {EVAL_DIR}")

    model_evals: List[ModelEval] = []
    for json_path in sorted(EVAL_DIR.glob("eval_*.json")):
        with open(json_path, "r", encoding="utf-8") as fp:
            blob = json.load(fp)

        tag = blob.get("tag") or json_path.stem
        results = blob.get("results") or {}
        stored_averages = blob.get("averages") or {}

        stored_score = stored_averages.get(PROP)

        method, k_val = infer_method(tag)
        model_evals.append(ModelEval(tag=tag, method=method, k=k_val, score=stored_score, source=json_path))

    if not model_evals:
        raise RuntimeError(f"No evaluation JSON files found in {EVAL_DIR}")

    return model_evals


def sort_key(entry: ModelEval) -> tuple[int, float, str]:
    if entry.method == "base":
        return (0, entry.k or 0, entry.tag)
    if entry.method == "vanilla":
        return (1, entry.k or 0, entry.tag)
    if entry.method == "top-k-tok":
        return (2, entry.k or math.inf, entry.tag)
    return (3, entry.k or math.inf, entry.tag)


def plot_model_performance(model_evals: List[ModelEval]) -> None:
    ordered = sorted(model_evals, key=sort_key)

    labels = [entry.label for entry in ordered]
    scores = [entry.score for entry in ordered]
    colors = []
    for entry in ordered:
        if entry.method == "base":
            colors.append("#4c72b0")  # blue
        elif entry.method == "vanilla":
            colors.append("#dd8452")  # orange
        elif entry.method == "top-k-tok":
            colors.append("#55a868")  # green
        else:
            colors.append("#c44e52")  # red

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(ordered)), scores, color=colors, edgecolor="#333333")

    ax.set_ylabel(f"Overall performance (\\texttt{{{PROP}}}) -- higher is better")
    ax.set_title("Light-suite evaluation summary")
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Annotate bars with the score value
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(scores) * 0.01,
            f"{score:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    handles = [
        plt.Line2D([0], [0], color="#4c72b0", lw=4, label="Base"),
        plt.Line2D([0], [0], color="#dd8452", lw=4, label="Vanilla Distillation"),
        plt.Line2D([0], [0], color="#55a868", lw=4, label="Entropy Top-k Distillation"),
    ]
    ax.legend(handles=handles, loc="upper left")

    fig.tight_layout()
    OUTPUT_PNG = "eval_summary.png"
    OUTPUT_PDF = "eval_summary.pdf"
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig("eval_summary.pgf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


model_evals = load_model_evals()
plot_model_performance(model_evals)

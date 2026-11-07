"""Plot the complete fixed-set comparison and its per-question outcomes."""

import json

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent


def main():
    summary = json.loads((ROOT / "results/summary.json").read_text())
    outputs = json.loads((ROOT / "results/heldout_outputs.json").read_text())
    review = json.loads((ROOT / "results/semantic_review.json").read_text())["rows"]
    verdict = {(r["condition"], r["id"]): r["content_pass"] for r in review}
    names = ["base", "prompt", "edited", "random", "reversed"]
    labels = [
        "Original",
        "Style prompt",
        "Analytical edit",
        "Random edit",
        "Reversed edit",
    ]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    assets = ROOT / "figures"
    assets.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3), gridspec_kw={"width_ratios": [2, 1]})
    x = np.arange(len(names))
    for offset, metric, label, color, marker in [
        (-0.18, "suffix_pass", "All sentences end in nyan", "#276FBF", "o"),
        (0, "content_pass", "Required content is correct", "#686868", "s"),
        (0.18, "joint_pass", "Both", "#A34400", "^"),
    ]:
        values = [summary[n][metric] * 5 for n in names]
        axes[0].scatter(x + offset, values, color=color, marker=marker, label=label, s=43)
    axes[0].set(
        xticks=x,
        xticklabels=labels,
        ylim=(-5, 105),
        ylabel="Prose cases passing (%)",
        title="A  Fixed prose set (n = 20)",
    )
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_yticks([0, 25, 50, 75, 100])
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=1,
        frameon=False,
        fontsize=8,
    )
    axes[1].barh(labels, [summary[n]["format_pass"] for n in names], color="#686868", height=0.6)
    axes[1].invert_yaxis()
    axes[1].set(
        xlim=(0, 4.5),
        xticks=[0, 1, 2, 3, 4],
        xlabel="Format cases passing (of 4)",
        title="B  Exact-format controls",
    )
    for i, n in enumerate(names):
        axes[1].text(
            summary[n]["format_pass"] + 0.07,
            i,
            str(summary[n]["format_pass"]),
            va="center",
        )
    fig.tight_layout()
    fig.savefig(assets / "comparison.svg", bbox_inches="tight")
    fig.savefig(assets / "comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    cases = [r["id"] for r in outputs if r["condition"] == "base" and r["kind"] == "prose"]
    matrix = []
    for name in names:
        line = []
        for case in cases:
            row = next(r for r in outputs if r["condition"] == name and r["id"] == case)
            state = 0
            if verdict[(name, case)]:
                state = 2 if row["all_sentences_nyan"] else 1
            if row["truncated"]:
                state = 3
            line.append(state)
        matrix.append(line)
    colors = ["#E6E6E6", "#8CA1B4", "#247B66", "#CF8B32"]
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.imshow(matrix, cmap=ListedColormap(colors), vmin=0, vmax=3, aspect="auto")
    ax.set(
        xticks=np.arange(len(cases)),
        xticklabels=cases,
        yticks=np.arange(len(names)),
        yticklabels=labels,
    )
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.set_xticks(np.arange(-0.5, len(cases)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    symbols = ["×", "C", "N", "!"]
    for row_index, line in enumerate(matrix):
        for column_index, state in enumerate(line):
            ax.text(
                column_index,
                row_index,
                symbols[state],
                ha="center",
                va="center",
                color="white" if state == 2 else "black",
                fontsize=9,
            )
    ax.legend(
        handles=[
            Patch(facecolor=c, label=label)
            for c, label in zip(
                colors,
                [
                    "× Content fails",
                    "C Content passes, style fails",
                    "N Content and style pass",
                    "! Token limit reached",
                ],
            )
        ],
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(assets / "case_matrix.svg", bbox_inches="tight")
    fig.savefig(assets / "case_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

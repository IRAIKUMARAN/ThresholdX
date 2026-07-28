"""
plots.py
========
Every figure the report uses. Kept separate from the analysis so that changing
how a chart looks can never change a number.

Uses the "Agg" backend, which draws straight to a file without needing a
screen - so this works the same when run over SSH or in a marking script.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

# One consistent palette across every figure.
PROPOSED_COLOUR = "#C1272D"
BASELINE_COLOUR = "#2E5EAA"
NEUTRAL_COLOUR  = "#7A7A7A"
ACCENT_COLOUR   = "#E8A33D"


def plot_convergence(history_4d, history_5d, save_path):
    """
    How the search improved over time, for both search spaces.

    The gap between the two lines is the visual argument of the whole project:
    the 4-D search flattens almost immediately because reweighting correlated
    models barely changes any decision, while the 5-D search keeps finding
    improvements because the threshold genuinely moves the cost.
    """
    figure, (cost_axis, temperature_axis) = plt.subplots(
        2, 1, figsize=(9, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    for history, label, colour in (
        (history_4d, "4-D search (weights only)", NEUTRAL_COLOUR),
        (history_5d, "5-D search (weights + threshold)", PROPOSED_COLOUR),
    ):
        if history is None:
            continue
        iterations = [step["iteration"] for step in history]
        best_costs = [step["best_cost"] for step in history]
        cost_axis.plot(iterations, best_costs, color=colour, linewidth=2, label=label)

    cost_axis.set_ylabel("Business cost  (5 x missed + 1 x false alarm)")
    cost_axis.set_title("Simulated Annealing convergence  (lower is better)",
                        fontweight="bold")
    cost_axis.legend()
    cost_axis.grid(alpha=0.3)

    # The temperature curve, so a reader can see the schedule really cools.
    if history_5d is not None:
        temperature_axis.plot(
            [step["iteration"] for step in history_5d],
            [step["temperature"] for step in history_5d],
            color=ACCENT_COLOUR, linewidth=1.5,
        )
    temperature_axis.set_ylabel("Temperature")
    temperature_axis.set_xlabel("Iteration")
    temperature_axis.set_yscale("log")
    temperature_axis.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    saved {save_path.name}")


def plot_cost_comparison(results, save_path):
    """
    Business cost for every system - the number the project is optimising.

    Sorted best-to-worst and drawn as a horizontal bar chart, because system
    names are long and this is the single most important figure in the report.
    """
    ordered = sorted(results, key=lambda row: row["Cost"])
    names   = [row["System"] for row in ordered]
    costs   = [row["Cost"] for row in ordered]

    colours = [
        PROPOSED_COLOUR if "SA " in name or "Proposed" in name else BASELINE_COLOUR
        for name in names
    ]

    figure, axis = plt.subplots(figsize=(9, 0.55 * len(names) + 1.5))
    bars = axis.barh(names, costs, color=colours, alpha=0.9)
    axis.bar_label(bars, fmt="%.0f", padding=4, fontsize=9)

    axis.set_xlabel("Business cost on the held-out test set  (lower is better)")
    axis.set_title("What each system would actually cost the business",
                   fontweight="bold")
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=0.3)
    axis.set_xlim(0, max(costs) * 1.15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    saved {save_path.name}")


def plot_metric_comparison(results, save_path):
    """F1 and recall side by side for every system."""
    names   = [row["System"] for row in results]
    f1s     = [row["F1"] for row in results]
    recalls = [row["Recall"] for row in results]

    positions = np.arange(len(names))
    width     = 0.38

    figure, axis = plt.subplots(figsize=(11, 5))
    bars_f1     = axis.bar(positions - width / 2, f1s, width,
                           label="F1", color=PROPOSED_COLOUR, alpha=0.9)
    bars_recall = axis.bar(positions + width / 2, recalls, width,
                           label="Recall", color=BASELINE_COLOUR, alpha=0.9)

    axis.bar_label(bars_f1, fmt="%.3f", padding=2, fontsize=7)
    axis.bar_label(bars_recall, fmt="%.3f", padding=2, fontsize=7)

    axis.set_ylabel("Score")
    axis.set_title("F1 and recall across all systems", fontweight="bold")
    axis.set_xticks(positions)
    axis.set_xticklabels(names, rotation=22, ha="right", fontsize=8)
    axis.set_ylim(0, 1.0)
    axis.legend()
    axis.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    saved {save_path.name}")


def plot_learned_weights(model_names, weights_4d, weights_5d, save_path):
    """
    How much the search decided to trust each classifier.

    Drawn as grouped bars rather than a pie chart. A pie chart cannot show two
    results side by side, and it renders a 0.1% slice as an unreadable sliver -
    which is exactly what happened to SVM in the original figure.
    """
    positions = np.arange(len(model_names))
    width     = 0.38

    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    bars_4d = axis.bar(positions - width / 2, weights_4d, width,
                       label="4-D search", color=NEUTRAL_COLOUR, alpha=0.9)
    bars_5d = axis.bar(positions + width / 2, weights_5d, width,
                       label="5-D search", color=PROPOSED_COLOUR, alpha=0.9)

    axis.bar_label(bars_4d, fmt="%.3f", padding=2, fontsize=8)
    axis.bar_label(bars_5d, fmt="%.3f", padding=2, fontsize=8)

    axis.set_ylabel("Weight  (trust placed in this model)")
    axis.set_title("Ensemble weights found by the search", fontweight="bold")
    axis.set_xticks(positions)
    axis.set_xticklabels(model_names, rotation=12, ha="right")
    axis.set_ylim(0, max(max(weights_4d), max(weights_5d)) * 1.25)
    axis.legend()
    axis.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    saved {save_path.name}")


def plot_threshold_curve(y_train, ensemble_probabilities, chosen_threshold, save_path):
    """
    Business cost as a function of the decision threshold.

    This figure exists to justify the central design decision of the project.
    It shows the cost curve is steep and its minimum sits well below 0.5, which
    is why freezing the cutoff at 0.5 - as the original code did everywhere -
    left a large amount of value on the table.
    """
    from src import cost as cost_module

    thresholds = np.linspace(config.MIN_THRESHOLD, config.MAX_THRESHOLD, 181)
    costs = [
        cost_module.business_cost_of_probabilities(y_train, ensemble_probabilities, t)
        for t in thresholds
    ]

    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    axis.plot(thresholds, costs, color=PROPOSED_COLOUR, linewidth=2)

    axis.axvline(0.5, color=NEUTRAL_COLOUR, linestyle="--", linewidth=1.5,
                 label="0.5  (the original hardcoded cutoff)")
    axis.axvline(chosen_threshold, color=ACCENT_COLOUR, linestyle="-", linewidth=1.8,
                 label=f"{chosen_threshold:.3f}  (found by the search)")

    axis.set_xlabel("Decision threshold")
    axis.set_ylabel("Business cost on training data")
    axis.set_title("Why the threshold matters more than the weights",
                   fontweight="bold")
    axis.legend()
    axis.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    saved {save_path.name}")

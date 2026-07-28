"""
run_pipeline.py
===============
Runs the whole project end to end.

    python run_pipeline.py              full run
    python run_pipeline.py --quick      fewer iterations, for a fast check
    python run_pipeline.py --seeds      also run the multi-seed robustness check

The six stages:

    1. Load and prepare the data
    2. Train the four base classifiers with out-of-fold predictions
    3. Search for the best ensemble - twice, 4-D and 5-D, so they can be compared
    4. Consult the Bayesian network on the uncertain customers
    5. Score every system on the held-out test set
    6. Draw the figures and save the results

A RULE THAT IS ENFORCED THROUGHOUT
----------------------------------
Every weight and every threshold is chosen using TRAINING data only. The test
set is touched exactly once, at stage 5, purely to report final numbers.
Choosing a threshold by looking at test results would be data leakage and would
invalidate the entire report.
"""

import argparse
import json

import numpy as np
import pandas as pd

import config
from src import annealing, base_models, baselines, bayes_net, data_prep, plots
from src import cost as cost_module


def banner(text):
    print()
    print("=" * 74)
    print(f"  {text}")
    print("=" * 74)


def stage(number, text):
    print()
    print(f"[{number}/6] {text}")
    print("-" * 74)


def run(quick=False, run_seed_check=False):
    max_iterations = 800 if quick else config.SA_MAX_ITERATIONS

    banner("ThresholdX - cost-aware ensemble search for customer churn")
    print("  ECE 569A  |  Grad Group 21")
    print(f"  Cost model: missing a churner = {config.COST_OF_MISSING_A_CHURNER:.0f}x "
          f"a false alarm")

    config.RESULTS_DIR.mkdir(exist_ok=True)

    # ---------------------------------------------------------------- 1
    stage(1, "Preparing the data")
    data = data_prep.prepare_data()

    # ---------------------------------------------------------------- 2
    stage(2, "Training the four base classifiers (out-of-fold)")
    oof_probabilities, trained_models, model_names = \
        base_models.train_out_of_fold(data["X_train"], data["y_train"])
    test_probabilities = base_models.predict_test_probabilities(
        trained_models, data["X_test"]
    )

    # ---------------------------------------------------------------- 3
    stage(3, "Searching for the best ensemble")

    print("\n  (a) Original design - weights only, cutoff frozen at 0.5")
    search_4d = annealing.simulated_annealing(
        oof_probabilities, data["y_train"],
        search_threshold=False, max_iterations=max_iterations,
    )

    print("\n  (b) Proposed design - weights and cutoff searched together")
    search_5d = annealing.simulated_annealing(
        oof_probabilities, data["y_train"],
        search_threshold=True, max_iterations=max_iterations,
    )

    blend_4d = search_4d["best_blend"]
    blend_5d = search_5d["best_blend"]

    print("\n  Weights found:")
    print(f"    {'Model':<22}{'4-D':>10}{'5-D':>10}")
    print("    " + "-" * 42)
    for index, name in enumerate(model_names):
        print(f"    {name:<22}{blend_4d.weights[index]:>10.3f}"
              f"{blend_5d.weights[index]:>10.3f}")
    print(f"    {'decision threshold':<22}{blend_4d.threshold:>10.3f}"
          f"{blend_5d.threshold:>10.3f}")

    # ---------------------------------------------------------------- 4
    stage(4, "Consulting the Bayesian network on uncertain customers")
    network = bayes_net.ChurnBayesNet()
    network.fit(data["raw_train"], data["y_train"])
    bayes_test_probabilities = network.predict_probability(data["raw_test"])

    ensemble_test_5d = blend_5d.predict_probabilities(test_probabilities)
    combined_test = bayes_net.apply_to_uncertain_cases(
        ensemble_test_5d, bayes_test_probabilities, blend_5d.threshold
    )

    # ---------------------------------------------------------------- 5
    stage(5, "Scoring every system on the held-out test set")
    results = []

    def score(name, train_probabilities, test_probs, threshold=None):
        """
        Add one system to the results table.

        If no threshold is given, one is tuned on TRAINING probabilities and
        then applied unchanged to the test set. Every baseline gets this same
        treatment, so the comparison stays fair in both directions.
        """
        if threshold is None:
            threshold, _ = cost_module.find_best_threshold(
                data["y_train"], train_probabilities
            )
        results.append(cost_module.evaluate(name, data["y_test"], test_probs, threshold))

    # The four classifiers on their own.
    for index, name in enumerate(model_names):
        score(name, oof_probabilities[:, index], test_probabilities[:, index])

    # The baseline the proposal names, now with a fair configuration.
    stack_train, stack_test = baselines.stacking_with_logistic_regression(
        oof_probabilities, data["y_train"], test_probabilities
    )
    score("LR Stacking (baseline)", stack_train, stack_test)

    # The no-learning control.
    uniform_train, uniform_test = baselines.uniform_ensemble(
        oof_probabilities, test_probabilities
    )
    score("Uniform Ensemble", uniform_train, uniform_test)

    # The two versions of the proposed system. Their thresholds come from the
    # search itself, so they are passed in explicitly rather than re-tuned.
    score("SA Ensemble 4-D (original)",
          None, blend_4d.predict_probabilities(test_probabilities),
          threshold=blend_4d.threshold)

    score("SA Ensemble 5-D (proposed)",
          None, ensemble_test_5d, threshold=blend_5d.threshold)

    score("SA 5-D + Bayesian Network",
          None, combined_test, threshold=blend_5d.threshold)

    results_table = pd.DataFrame(results)

    display = results_table[
        ["System", "Threshold", "Precision", "Recall", "F1", "AUC-ROC", "FN", "FP", "Cost"]
    ].copy()
    for column in ["Precision", "Recall", "F1", "AUC-ROC"]:
        display[column] = display[column].map("{:.3f}".format)

    print()
    print(display.to_string(index=False))

    # ---------------------------------------------------------------- 6
    stage(6, "Drawing figures and saving results")

    plots.plot_convergence(search_4d["history"], search_5d["history"],
                           config.RESULTS_DIR / "fig1_convergence.png")
    plots.plot_cost_comparison(results, config.RESULTS_DIR / "fig2_cost.png")
    plots.plot_metric_comparison(results, config.RESULTS_DIR / "fig3_metrics.png")
    plots.plot_learned_weights(model_names, blend_4d.weights, blend_5d.weights,
                               config.RESULTS_DIR / "fig4_weights.png")
    plots.plot_threshold_curve(
        data["y_train"],
        blend_5d.predict_probabilities(oof_probabilities),
        blend_5d.threshold,
        config.RESULTS_DIR / "fig5_threshold.png",
    )

    results_table.to_csv(config.RESULTS_DIR / "results_table.csv", index=False)
    print(f"    saved results_table.csv")

    # ------------------------------------------------------------ verdict
    banner("What the numbers say")
    verdict = summarise(results_table, model_names)

    # -------------------------------------------------- robustness check
    seed_summary = None
    if run_seed_check:
        seed_summary = robustness_check(oof_probabilities, data["y_train"],
                                        test_probabilities, data["y_test"],
                                        max_iterations)

    run_record = {
        "results": results_table.to_dict(orient="records"),
        "weights_4d": dict(zip(model_names, blend_4d.weights.round(4).tolist())),
        "weights_5d": dict(zip(model_names, blend_5d.weights.round(4).tolist())),
        "threshold_5d": round(blend_5d.threshold, 4),
        "sa_acceptance_rate_5d": round(search_5d["acceptance_rate"], 4),
        "verdict": verdict,
        "robustness": seed_summary,
    }
    with open(config.RESULTS_DIR / "run_record.json", "w") as handle:
        json.dump(run_record, handle, indent=2, default=float)

    print(f"\n  Everything written to {config.RESULTS_DIR}")
    return results_table


def summarise(results_table, model_names):
    """
    State plainly whether the proposed system beat what it needed to beat.

    Written to report a disappointing result just as clearly as a good one.
    A comparison that can only ever come out favourable is not evidence.
    """
    def cost_of(system_name):
        row = results_table[results_table["System"] == system_name]
        return float(row["Cost"].values[0])

    proposed = cost_of("SA Ensemble 5-D (proposed)")
    original = cost_of("SA Ensemble 4-D (original)")
    stacking = cost_of("LR Stacking (baseline)")

    single_model_costs = {name: cost_of(name) for name in model_names}
    best_single_name   = min(single_model_costs, key=single_model_costs.get)
    best_single_cost   = single_model_costs[best_single_name]

    def compare(label, other_cost):
        difference = other_cost - proposed
        percent    = 100 * difference / other_cost if other_cost else 0.0
        verdict    = "better" if difference > 0 else "WORSE"
        print(f"  vs {label:<34} {proposed:>6.0f} against {other_cost:>6.0f}   "
              f"{difference:+6.0f}  ({percent:+.1f}%)  {verdict}")
        return {"baseline_cost": other_cost, "difference": difference,
                "percent": percent}

    print("  Business cost of the proposed system versus each alternative:")
    print()
    against_stacking = compare("LR Stacking (the named baseline)", stacking)
    against_original = compare("its own 4-D original design", original)
    against_single   = compare(f"{best_single_name} alone", best_single_cost)

    print()
    if against_single["difference"] <= 0:
        print("  HONEST READING: the proposed system does NOT beat the best single")
        print("  classifier. The ensemble machinery is not earning its place, and")
        print("  the report should say so rather than compare only against stacking.")
    elif against_single["percent"] < 2.0:
        print("  HONEST READING: the gain over the best single classifier is under")
        print("  2%, which is within the range that run-to-run randomness can")
        print("  produce. Run with --seeds before claiming this as a result.")
    else:
        print(f"  The proposed system beats every alternative, including "
              f"{best_single_name}")
        print("  on its own - which is the comparison that actually matters.")

    return {
        "proposed_cost": proposed,
        "vs_stacking": against_stacking,
        "vs_original_4d": against_original,
        "vs_best_single": {**against_single, "model": best_single_name},
    }


def robustness_check(oof_probabilities, y_train, test_probabilities, y_test,
                     max_iterations):
    """
    Repeat the search with different random seeds.

    Simulated annealing is a randomised algorithm, so a single run tells you
    almost nothing. If the spread across seeds is wider than the margin over
    the baseline, then the margin is noise.
    """
    banner("Robustness check across random seeds")

    costs      = []
    thresholds = []

    for seed in config.ROBUSTNESS_SEEDS:
        search = annealing.simulated_annealing(
            oof_probabilities, y_train,
            search_threshold=True, max_iterations=max_iterations,
            random_seed=seed, verbose=False,
        )
        blend = search["best_blend"]
        test_cost = cost_module.business_cost_of_probabilities(
            y_test, blend.predict_probabilities(test_probabilities), blend.threshold
        )
        costs.append(test_cost)
        thresholds.append(blend.threshold)
        print(f"  seed {seed:>5}   threshold {blend.threshold:.3f}   "
              f"test cost {test_cost:.0f}")

    costs = np.array(costs)
    print()
    print(f"  mean test cost {costs.mean():.1f}  "
          f"standard deviation {costs.std():.1f}  "
          f"range {costs.min():.0f} to {costs.max():.0f}")

    return {
        "seeds": config.ROBUSTNESS_SEEDS,
        "costs": costs.tolist(),
        "thresholds": [round(t, 4) for t in thresholds],
        "mean": float(costs.mean()),
        "std": float(costs.std()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ThresholdX pipeline.")
    parser.add_argument("--quick", action="store_true",
                        help="fewer search iterations, for a fast sanity check")
    parser.add_argument("--seeds", action="store_true",
                        help="also repeat the search across several random seeds")
    arguments = parser.parse_args()

    run(quick=arguments.quick, run_seed_check=arguments.seeds)

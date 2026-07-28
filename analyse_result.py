"""
analyse_result.py
=================
Answers the question the main pipeline raises but cannot settle:

    The proposed system loses to Logistic Regression on its own. Why?

    python analyse_result.py             the three core experiments
    python analyse_result.py --splits 5  also repeat across 5 train/test splits
                                         (slow - retrains everything each time)

There are three candidate explanations, and each experiment below rules one in
or out. This is the part of the project that turns "our method lost" into a
finding worth reporting.
"""

import argparse

import numpy as np
import pandas as pd

import config
from src import annealing, base_models, data_prep
from src import cost as cost_module


def heading(text):
    print()
    print("=" * 74)
    print(f"  {text}")
    print("=" * 74)


def subheading(text):
    print()
    print(text)
    print("-" * 74)


# ---------------------------------------------------------------------------
# Experiment 1 - where does the improvement actually come from?
# ---------------------------------------------------------------------------

def decompose_the_gain(oof_probabilities, y_train, test_probabilities, y_test,
                       sa_weights, model_names):
    """
    The project has two levers: the ensemble WEIGHTS and the decision
    THRESHOLD. This separates their contributions by switching each on
    independently.

        equal weights  + threshold 0.5    <- the naive starting point
        equal weights  + tuned threshold  <- threshold alone
        SA weights     + threshold 0.5    <- weights alone
        SA weights     + tuned threshold  <- both

    If almost all the gain appears in the "threshold alone" row, then the
    ensemble search - the centrepiece of the proposal - is not what is doing
    the work, and the report has to say so.
    """
    subheading("EXPERIMENT 1  -  which lever produces the gain?")

    equal_weights = np.ones(len(model_names)) / len(model_names)

    def blend(weights, probabilities):
        return probabilities @ weights

    rows = []
    for label, weights in (("equal weights", equal_weights),
                           ("SA weights", sa_weights)):
        for threshold_mode in ("fixed at 0.5", "tuned on training"):
            train_blend = blend(weights, oof_probabilities)
            test_blend  = blend(weights, test_probabilities)

            if threshold_mode == "fixed at 0.5":
                threshold = config.DEFAULT_THRESHOLD
            else:
                threshold, _ = cost_module.find_best_threshold(y_train, train_blend)

            rows.append({
                "weights":   label,
                "threshold": threshold_mode,
                "value":     round(threshold, 3),
                "train cost": cost_module.business_cost_of_probabilities(
                    y_train, train_blend, threshold),
                "test cost": cost_module.business_cost_of_probabilities(
                    y_test, test_blend, threshold),
            })

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    naive     = table.iloc[0]["test cost"]
    threshold_only = table.iloc[1]["test cost"]
    weights_only   = table.iloc[2]["test cost"]
    both           = table.iloc[3]["test cost"]

    gain_from_threshold = naive - threshold_only
    gain_from_weights   = naive - weights_only
    gain_from_both      = naive - both

    print()
    print(f"  Starting from equal weights at a 0.5 cutoff (test cost {naive:.0f}):")
    print(f"    tuning the threshold alone saves   {gain_from_threshold:>6.0f}")
    print(f"    optimising the weights alone saves {gain_from_weights:>6.0f}")
    print(f"    doing both saves                   {gain_from_both:>6.0f}")

    if gain_from_both > 0:
        share = 100 * gain_from_threshold / gain_from_both
        print()
        print(f"  The threshold accounts for {share:.0f}% of the total improvement.")

    return table


# ---------------------------------------------------------------------------
# Experiment 2 - are the base models too similar to be worth blending?
# ---------------------------------------------------------------------------

def measure_model_agreement(oof_probabilities, model_names):
    """
    An ensemble only helps when its members make DIFFERENT mistakes. If all
    four classifiers agree, averaging them just reproduces one of them.

    This is the explanation offered in the README for why the 4-D search was
    nearly flat, so it needs a number attached rather than being asserted.
    """
    subheading("EXPERIMENT 2  -  how different are the four classifiers?")

    correlations = np.corrcoef(oof_probabilities.T)
    table = pd.DataFrame(correlations, index=model_names, columns=model_names)
    print(table.round(3).to_string())

    off_diagonal = correlations[np.triu_indices(len(model_names), k=1)]
    print()
    print(f"  Average correlation between models: {off_diagonal.mean():.3f}")
    print(f"  Highest pair: {off_diagonal.max():.3f}    "
          f"Lowest pair: {off_diagonal.min():.3f}")
    print()
    if off_diagonal.mean() > 0.9:
        print("  These models are near-duplicates of each other. Blending")
        print("  near-duplicates cannot produce much, which is the structural")
        print("  reason the weight search has so little to find.")
    elif off_diagonal.mean() > 0.75:
        print("  The models agree closely. There is some diversity to exploit,")
        print("  but not much.")
    else:
        print("  The models are genuinely diverse, so low ensemble gain is NOT")
        print("  explained by redundancy - look to overfitting instead.")

    return off_diagonal.mean()


# ---------------------------------------------------------------------------
# Experiment 3 - is the search overfitting the training set?
# ---------------------------------------------------------------------------

def measure_overfitting(oof_probabilities, y_train, test_probabilities, y_test,
                        sa_weights, model_names):
    """
    Compare each system's TRAINING cost against its TEST cost.

    Simulated annealing tunes 5 numbers against the training set. Logistic
    regression on its own tunes 1 - just its threshold. More free parameters
    means more chance to fit patterns that do not survive to new customers.

    The signature of overfitting is a system that wins on training data and
    loses on test data. If the search shows that pattern, the honest
    conclusion is that the extra parameters cost more than they earn.
    """
    subheading("EXPERIMENT 3  -  does the search overfit the training data?")

    systems = {}

    for column, name in enumerate(model_names):
        systems[name] = (oof_probabilities[:, column],
                         test_probabilities[:, column], 1)

    equal_weights = np.ones(len(model_names)) / len(model_names)
    systems["Uniform Ensemble"] = (oof_probabilities @ equal_weights,
                                   test_probabilities @ equal_weights, 1)
    systems["SA Ensemble 5-D"] = (oof_probabilities @ sa_weights,
                                  test_probabilities @ sa_weights,
                                  len(model_names) + 1)

    rows = []
    for name, (train_blend, test_blend, free_parameters) in systems.items():
        threshold, train_cost = cost_module.find_best_threshold(y_train, train_blend)
        test_cost = cost_module.business_cost_of_probabilities(
            y_test, test_blend, threshold)

        # Costs are on different-sized sets, so scale to cost per customer.
        train_per_customer = train_cost / len(y_train)
        test_per_customer  = test_cost / len(y_test)

        rows.append({
            "System": name,
            "free params": free_parameters,
            "train cost/cust": round(train_per_customer, 4),
            "test cost/cust":  round(test_per_customer, 4),
            "gap": round(test_per_customer - train_per_customer, 4),
            "test cost": test_cost,
        })

    table = pd.DataFrame(rows).sort_values("test cost")
    print(table.to_string(index=False))

    search_row = table[table["System"] == "SA Ensemble 5-D"].iloc[0]
    best_row   = table.iloc[0]

    print()
    print(f"  Lowest TRAINING cost per customer: "
          f"{table.loc[table['train cost/cust'].idxmin(), 'System']}")
    print(f"  Lowest TEST cost per customer:     {best_row['System']}")
    print()

    if table["train cost/cust"].idxmin() == search_row.name and \
            best_row["System"] != "SA Ensemble 5-D":
        print("  The search wins on training data and loses on test data.")
        print("  That is overfitting: its five free parameters latched onto")
        print("  patterns in the training set that did not survive.")
    else:
        print("  The search does not show the classic overfitting signature")
        print("  here - it is not simply memorising the training set.")

    return table


# ---------------------------------------------------------------------------
# Experiment 4 - does the conclusion survive a different train/test split?
# ---------------------------------------------------------------------------

def repeat_across_splits(split_count):
    """
    Everything so far rests on ONE 80/20 split of the customers. A conclusion
    that only holds for one split is not a conclusion.

    This repeats the entire pipeline - preprocessing, out-of-fold training,
    annealing - on several different splits and reports how often each system
    wins. Slow, because every split retrains all four classifiers.
    """
    heading(f"EXPERIMENT 4  -  repeating across {split_count} train/test splits")
    print("  (this retrains everything each time and takes a few minutes)")

    split_seeds = [42, 7, 123, 2024, 999, 31, 77, 2025][:split_count]
    all_rows = []

    for split_number, seed in enumerate(split_seeds, start=1):
        data = data_prep.prepare_data(split_seed=seed, quiet=True)
        oof, trained, names = base_models.train_out_of_fold(
            data["X_train"], data["y_train"], quiet=True)
        test_probabilities = base_models.predict_test_probabilities(
            trained, data["X_test"])

        search = annealing.simulated_annealing(
            oof, data["y_train"], search_threshold=True, verbose=False)
        weights = search["best_blend"].weights

        row = {"split": seed}

        for column, name in enumerate(names):
            threshold, _ = cost_module.find_best_threshold(
                data["y_train"], oof[:, column])
            row[name] = cost_module.business_cost_of_probabilities(
                data["y_test"], test_probabilities[:, column], threshold)

        row["SA 5-D"] = cost_module.business_cost_of_probabilities(
            data["y_test"], test_probabilities @ weights,
            search["best_blend"].threshold)

        all_rows.append(row)
        winner = min((k for k in row if k != "split"), key=lambda k: row[k])
        print(f"  split {split_number}/{split_count} (seed {seed:>4}) "
              f"- best system: {winner}")

    table = pd.DataFrame(all_rows).set_index("split")
    print()
    print("  Test cost by split:")
    print(table.to_string())

    print()
    print("  Mean test cost across splits:")
    means = table.mean().sort_values()
    for name, value in means.items():
        print(f"    {name:<24}{value:>8.1f}")

    win_counts = table.idxmin(axis=1).value_counts()
    print()
    print("  Times each system was cheapest:")
    for name, count in win_counts.items():
        print(f"    {name:<24}{count:>3} of {len(table)}")

    best_overall = means.index[0]
    print()
    if best_overall == "SA 5-D":
        print("  The proposed system is best on average across splits.")
    else:
        print(f"  {best_overall} is cheapest on average across splits, not the")
        print("  proposed system. This is not an artefact of one unlucky split.")

    return table


# ---------------------------------------------------------------------------

def test_the_overfitting_guard(oof_probabilities, y_train,
                               test_probabilities, y_test):
    """
    Does holding back a validation slice actually reduce overfitting?

    The search tunes five numbers, so it could in principle latch onto patterns
    that exist only in the training data. The textbook remedy is to hold back
    a slice the search cannot see and select the final solution on that.

    We implemented that remedy and it made things WORSE on a controlled
    benchmark, for two reasons worth understanding:

      1. The search evaluates thousands of candidate solutions. Picking
         whichever scores best on a held-back slice means fitting that slice.
         The guard against overfitting becomes a fresh way to overfit.
      2. Holding data back shrinks what the search explores, making its
         objective noisier.

    This experiment settles it on the real dataset instead of a benchmark. It
    runs both settings across several seeds and compares TEST cost, which
    neither setting has ever seen.
    """
    subheading("EXPERIMENT 5  -  does a validation guard reduce overfitting?")

    results = {}

    for use_guard in (False, True):
        costs = []
        for seed in config.ROBUSTNESS_SEEDS:
            search = annealing.simulated_annealing(
                oof_probabilities, y_train, search_threshold=True,
                random_seed=seed, use_validation=use_guard, verbose=False)
            blend = search["best_blend"]
            costs.append(cost_module.business_cost_of_probabilities(
                y_test, test_probabilities @ blend.weights, blend.threshold))

        results["guard on" if use_guard else "guard off"] = np.array(costs)

    print(f"  {'Setting':<12}{'mean':>9}{'std':>8}{'min':>7}{'max':>7}")
    print("  " + "-" * 44)
    for label, costs in results.items():
        print(f"  {label:<12}{costs.mean():>9.1f}{costs.std():>8.1f}"
              f"{costs.min():>7.0f}{costs.max():>7.0f}")

    off = results["guard off"].mean()
    on  = results["guard on"].mean()

    print()
    if on < off:
        print(f"  The guard helps on this dataset: {on:.1f} against {off:.1f}.")
        print("  Consider setting SA_USE_VALIDATION = True in config.py.")
    else:
        print(f"  The guard does NOT help here: {on:.1f} against {off:.1f}.")
        print("  This matches the benchmark result, and is why it ships off.")
        print("  The search is not overfitting badly enough for the remedy to")
        print("  be worth the data it costs.")

    return results


def main(split_count):
    heading("Why does the proposed system lose? - diagnostic analysis")

    print()
    print("  Preparing data and training base models (about a minute)")
    data = data_prep.prepare_data(quiet=True)
    oof_probabilities, trained_models, model_names = \
        base_models.train_out_of_fold(data["X_train"], data["y_train"], quiet=True)
    test_probabilities = base_models.predict_test_probabilities(
        trained_models, data["X_test"])

    search = annealing.simulated_annealing(
        oof_probabilities, data["y_train"], search_threshold=True, verbose=False)
    sa_weights = search["best_blend"].weights

    print(f"  SA weights: " + ", ".join(
        f"{name} {weight:.3f}" for name, weight in zip(model_names, sa_weights)))
    print(f"  SA threshold: {search['best_blend'].threshold:.3f}")

    decompose_the_gain(oof_probabilities, data["y_train"],
                       test_probabilities, data["y_test"],
                       sa_weights, model_names)

    measure_model_agreement(oof_probabilities, model_names)

    measure_overfitting(oof_probabilities, data["y_train"],
                        test_probabilities, data["y_test"],
                        sa_weights, model_names)

    test_the_overfitting_guard(oof_probabilities, data["y_train"],
                               test_probabilities, data["y_test"])

    if split_count:
        repeat_across_splits(split_count)

    heading("Done")
    print("  Use these three experiments to write the discussion section.")
    print("  A negative result that is explained and evidenced is a stronger")
    print("  contribution than a positive one that is not.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=int, default=0, metavar="N",
                        help="also repeat the comparison across N train/test splits")
    arguments = parser.parse_args()
    main(arguments.splits)

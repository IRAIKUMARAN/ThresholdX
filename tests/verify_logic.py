"""
verify_logic.py
===============
Checks the parts of the project that carry the argument.

Run it with:      python tests/verify_logic.py

These are not decorative tests. Each one targets a specific claim the report
makes, or a specific bug that was found in the original code. If a test here
fails, a sentence in the report is wrong.

None of these need scikit-learn, so they run anywhere.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

import config
from src import annealing, bayes_net
from src import cost as cost_module


passed = 0
failed = 0


def check(description, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {description}")
    else:
        failed += 1
        print(f"  FAIL  {description}")
        if detail:
            print(f"        {detail}")


def section(title):
    print()
    print(title)
    print("-" * 70)


# ---------------------------------------------------------------------------
section("1. The cost function counts the right things")
# ---------------------------------------------------------------------------

truth      = np.array([1, 1, 0, 0, 1])
prediction = np.array([1, 0, 1, 0, 0])
# caught 1 churner, missed 2, one false alarm

counts = cost_module.count_confusion(truth, prediction)
check("true positives counted", counts["true_positives"] == 1, str(counts))
check("false negatives counted", counts["false_negatives"] == 2, str(counts))
check("false positives counted", counts["false_positives"] == 1, str(counts))
check("true negatives counted", counts["true_negatives"] == 1, str(counts))

# 2 misses at 5 each, plus 1 false alarm at 1 = 11
check("business cost = 5*misses + 1*false alarms",
      cost_module.business_cost(truth, prediction) == 11.0,
      f"got {cost_module.business_cost(truth, prediction)}")

check("a perfect prediction costs nothing",
      cost_module.business_cost(truth, truth) == 0.0)

check("missing a churner really is 5x worse than a false alarm",
      cost_module.business_cost(np.array([1]), np.array([0])) ==
      5 * cost_module.business_cost(np.array([0]), np.array([1])))


# ---------------------------------------------------------------------------
section("2. Metrics match their textbook definitions")
# ---------------------------------------------------------------------------

precision, recall, f1 = cost_module.precision_recall_f1(truth, prediction)
check("precision = TP/(TP+FP) = 1/2", abs(precision - 0.5) < 1e-9, f"got {precision}")
check("recall    = TP/(TP+FN) = 1/3", abs(recall - 1/3) < 1e-9, f"got {recall}")
check("f1 = harmonic mean of the two", abs(f1 - 0.4) < 1e-9, f"got {f1}")

# AUC checked against a brute-force count over every churner/non-churner pair.
rng = np.random.default_rng(0)
labels_sample = rng.integers(0, 2, 300)
scores_sample = rng.random(300)

wins = ties = total = 0
for churner_score in scores_sample[labels_sample == 1]:
    for other_score in scores_sample[labels_sample == 0]:
        total += 1
        if churner_score > other_score:
            wins += 1
        elif churner_score == other_score:
            ties += 1
brute_force_auc = (wins + 0.5 * ties) / total

check("AUC matches brute-force pairwise definition",
      abs(cost_module.area_under_roc(labels_sample, scores_sample) - brute_force_auc) < 1e-9,
      f"ours {cost_module.area_under_roc(labels_sample, scores_sample):.6f} "
      f"vs brute force {brute_force_auc:.6f}")

check("a perfect ranking scores AUC 1.0",
      abs(cost_module.area_under_roc(np.array([0, 0, 1, 1]),
                                     np.array([0.1, 0.2, 0.8, 0.9])) - 1.0) < 1e-9)


# ---------------------------------------------------------------------------
section("3. Threshold tuning genuinely finds the cheapest cutoff")
# ---------------------------------------------------------------------------

labels = (rng.random(3000) < 0.27).astype(int)
probabilities = np.clip(0.2 + 0.4 * labels + rng.normal(0, 0.2, 3000), 0.01, 0.99)

best_threshold, best_cost = cost_module.find_best_threshold(labels, probabilities)
cost_at_half = cost_module.business_cost_of_probabilities(labels, probabilities, 0.5)

check("tuned threshold costs no more than the default 0.5",
      best_cost <= cost_at_half, f"tuned {best_cost} vs 0.5 -> {cost_at_half}")

# Nothing in the grid should beat what the search returned.
grid = np.linspace(config.MIN_THRESHOLD, config.MAX_THRESHOLD, 181)
grid_best = min(
    cost_module.business_cost_of_probabilities(labels, probabilities, t) for t in grid
)
check("no cutoff in the grid beats the one returned",
      best_cost <= grid_best + 1e-9)

check("because misses cost 5x, the best cutoff sits below 0.5",
      best_threshold < 0.5,
      f"best threshold {best_threshold:.3f} - if this is above 0.5 the "
      f"cost asymmetry is not being applied")


# ---------------------------------------------------------------------------
section("4. Simulated annealing behaves like annealing, not greedy search")
# ---------------------------------------------------------------------------

n = 2500
labels = (rng.random(n) < 0.265).astype(int)
signal = np.clip(0.25 + 0.45 * labels + rng.normal(0, 0.22, n), 0.01, 0.99)
probability_matrix = np.clip(
    np.column_stack([signal + rng.normal(0, s, n) for s in (0.06, 0.09, 0.07, 0.15)]),
    0.01, 0.99,
)

search_5d = annealing.simulated_annealing(
    probability_matrix, labels, search_threshold=True,
    max_iterations=1200, verbose=False,
)
search_4d = annealing.simulated_annealing(
    probability_matrix, labels, search_threshold=False,
    max_iterations=1200, verbose=False,
)

# The original bug: acceptance rate was 1.2%, meaning uphill moves were never
# taken and the algorithm was really just hill-climbing.
check("uphill moves are actually being accepted (the temperature bug)",
      search_5d["uphill_acceptance_rate"] > 0.02,
      f"uphill acceptance {search_5d['uphill_acceptance_rate']:.1%} - "
      f"below 2% means the temperature is still mis-scaled")

check("temperature genuinely cools during the run",
      search_5d["end_temperature"] < search_5d["start_temperature"],
      f"{search_5d['start_temperature']:.2f} -> {search_5d['end_temperature']:.2f}")

check("the full iteration budget is used (the early-stop bug)",
      len(search_5d["history"]) == 1200,
      f"ran {len(search_5d['history'])} of 1200 iterations")

check("weights stay on the simplex - they sum to 1",
      abs(search_5d["best_blend"].weights.sum() - 1.0) < 1e-9)

check("weights never go negative",
      np.all(search_5d["best_blend"].weights >= 0))

check("the threshold stays inside its allowed range",
      config.MIN_THRESHOLD <= search_5d["best_blend"].threshold <= config.MAX_THRESHOLD)

check("the search never ends worse than where it started",
      search_5d["best_cost"] <= cost_module.business_cost_of_probabilities(
          labels, probability_matrix @ (np.ones(4) / 4), 0.5))

check("searching the threshold beats freezing it at 0.5",
      search_5d["best_cost"] <= search_4d["best_cost"],
      f"5-D {search_5d['best_cost']:.0f} vs 4-D {search_4d['best_cost']:.0f}")

check("the same seed reproduces the same answer",
      annealing.simulated_annealing(
          probability_matrix, labels, max_iterations=300,
          random_seed=1, verbose=False)["best_cost"] ==
      annealing.simulated_annealing(
          probability_matrix, labels, max_iterations=300,
          random_seed=1, verbose=False)["best_cost"])


# ---------------------------------------------------------------------------
section("5. The Bayesian network - the two bugs that were found")
# ---------------------------------------------------------------------------

customers = pd.read_csv(config.DATA_FILE)
customers["TotalCharges"] = pd.to_numeric(
    customers["TotalCharges"].replace(" ", np.nan), errors="coerce").fillna(0.0)
customers["Churned"] = (customers["Churn"] == "Yes").astype(int)
churn_labels = customers["Churned"].values

facts = bayes_net.describe_customers(customers)

# BUG: the old code read Contract from one-hot columns that did not exist in
# the raw table, so every customer silently became "Month-to-month".
check("contract type carries real information, not one constant value",
      facts["Contract"].nunique() == 3,
      f"found {facts['Contract'].nunique()} distinct contract values: "
      f"{list(facts['Contract'].unique())}")

check("every evidence variable has more than one state",
      all(facts[column].nunique() > 1 for column in bayes_net.EVIDENCE_COLUMNS),
      str({c: facts[c].nunique() for c in bayes_net.EVIDENCE_COLUMNS}))

check("no customer ends up with a missing fact",
      not facts.isin(["nan", "NaN"]).any().any())

# BUG: the old network's decisive tables were hardcoded, so fit() had no
# effect whatsoever on its predictions.
train_rows = slice(0, 5000)
test_rows  = slice(5000, None)

network = bayes_net.ChurnBayesNet().fit(
    customers.iloc[train_rows], churn_labels[train_rows])
predictions_real = network.predict_probability(customers.iloc[test_rows])

scrambled = rng.permutation(churn_labels[train_rows])
network_scrambled = bayes_net.ChurnBayesNet().fit(
    customers.iloc[train_rows], scrambled)
predictions_scrambled = network_scrambled.predict_probability(customers.iloc[test_rows])

check("training on scrambled labels changes the output (fit() does something)",
      not np.allclose(predictions_real, predictions_scrambled),
      "identical output means the probability tables are not learned from data")

check("real labels predict better than scrambled ones",
      cost_module.area_under_roc(churn_labels[test_rows], predictions_real) >
      cost_module.area_under_roc(churn_labels[test_rows], predictions_scrambled))

check("output is a valid probability for every customer",
      np.all((predictions_real >= 0) & (predictions_real <= 1)))

check("predicted churn rate is close to the true churn rate",
      abs(predictions_real.mean() - churn_labels[test_rows].mean()) < 0.05,
      f"predicted {predictions_real.mean():.3f} "
      f"vs actual {churn_labels[test_rows].mean():.3f}")

check("the network is better than guessing (AUC above 0.5)",
      cost_module.area_under_roc(churn_labels[test_rows], predictions_real) > 0.70,
      f"AUC {cost_module.area_under_roc(churn_labels[test_rows], predictions_real):.3f}")

# Mismatched lengths must raise rather than silently misalign - this is the
# guard against the original label-alignment bug.
try:
    bayes_net.ChurnBayesNet().fit(customers.iloc[:100], churn_labels[:50])
    check("mismatched customers and labels are rejected", False,
          "no error was raised - misalignment could pass unnoticed")
except ValueError:
    check("mismatched customers and labels are rejected", True)


# ---------------------------------------------------------------------------
section("6. The uncertainty band follows the threshold")
# ---------------------------------------------------------------------------

ensemble = np.array([0.05, 0.30, 0.35, 0.40, 0.75, 0.95])
network_opinion = np.full(6, 0.99)

adjusted = bayes_net.apply_to_uncertain_cases(
    ensemble, network_opinion, threshold=0.35, band=0.10)

check("confident predictions are left untouched",
      adjusted[0] == 0.05 and adjusted[5] == 0.95)
check("uncertain predictions are handed to the network",
      adjusted[2] == 0.99 and adjusted[3] == 0.99)
check("the band moves with the threshold rather than sitting at 0.5",
      adjusted[4] == 0.75,
      "0.75 is far from a 0.35 cutoff and should not have been overridden")


# ---------------------------------------------------------------------------
print()
print("=" * 70)
print(f"  {passed} passed, {failed} failed")
print("=" * 70)

sys.exit(1 if failed else 0)

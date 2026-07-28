"""
cost.py
=======
The business cost function, and the metrics we report.

This file is deliberately tiny and has no machine-learning dependencies.
It answers one question: given the true answers and our predictions,
how much money did we lose?

Everything else in the project - the annealing search, the ensemble, the
Bayesian network - exists only to make the number this file computes smaller.
"""

import numpy as np

import config


# ---------------------------------------------------------------------------
# Turning probabilities into decisions
# ---------------------------------------------------------------------------

def probabilities_to_predictions(churn_probabilities, threshold):
    """
    Convert probabilities into yes/no churn predictions.

    A customer is flagged as "will churn" when their probability is at or
    above the threshold.
    """
    return (churn_probabilities >= threshold).astype(int)


def count_confusion(true_labels, predicted_labels):
    """
    Count the four possible outcomes.

    Returns a dict with:
      true_positives  - churner, correctly caught
      false_positives - loyal customer, wrongly flagged  (cheap mistake)
      true_negatives  - loyal customer, correctly left alone
      false_negatives - churner we MISSED                (expensive mistake)

    Written out with explicit boolean masks rather than a library call so the
    definition of each term is visible on the page.
    """
    actually_churned = (true_labels == 1)
    we_said_churn    = (predicted_labels == 1)

    return {
        "true_positives":  int(np.sum(actually_churned  & we_said_churn)),
        "false_positives": int(np.sum(~actually_churned & we_said_churn)),
        "true_negatives":  int(np.sum(~actually_churned & ~we_said_churn)),
        "false_negatives": int(np.sum(actually_churned  & ~we_said_churn)),
    }


# ---------------------------------------------------------------------------
# The business cost - the objective of this entire project
# ---------------------------------------------------------------------------

def business_cost(true_labels, predicted_labels):
    """
    Total business cost = 5 * (missed churners) + 1 * (false alarms).

    LOWER IS BETTER. This is a cost, not a score.

    The 5 comes from config.COST_OF_MISSING_A_CHURNER and encodes the fact
    that losing a customer hurts far more than wasting a retention offer.
    """
    counts = count_confusion(true_labels, predicted_labels)

    cost_of_misses      = counts["false_negatives"] * config.COST_OF_MISSING_A_CHURNER
    cost_of_false_alarms = counts["false_positives"] * config.COST_OF_FALSE_ALARM

    return cost_of_misses + cost_of_false_alarms


def business_cost_of_probabilities(true_labels, churn_probabilities, threshold):
    """Convenience wrapper: threshold the probabilities, then score them."""
    predictions = probabilities_to_predictions(churn_probabilities, threshold)
    return business_cost(true_labels, predictions)


# ---------------------------------------------------------------------------
# Standard classification metrics
# ---------------------------------------------------------------------------
# Implemented directly from their definitions rather than imported, so that
# the report can state exactly what was computed.

def precision_recall_f1(true_labels, predicted_labels):
    """
    precision - of the customers we flagged, what fraction really churned?
    recall    - of the customers who really churned, what fraction did we catch?
    f1        - the harmonic mean of the two

    Recall is the one that matters most here, because a missed churner is the
    expensive mistake.
    """
    counts = count_confusion(true_labels, predicted_labels)

    true_positives  = counts["true_positives"]
    false_positives = counts["false_positives"]
    false_negatives = counts["false_negatives"]

    flagged_count  = true_positives + false_positives
    churner_count  = true_positives + false_negatives

    precision = true_positives / flagged_count if flagged_count > 0 else 0.0
    recall    = true_positives / churner_count if churner_count > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def area_under_roc(true_labels, churn_probabilities):
    """
    AUC-ROC, computed via its rank interpretation:

        AUC = probability that a randomly chosen churner is scored higher
              than a randomly chosen non-churner.

    Unlike the metrics above, AUC does not depend on the threshold at all.
    That makes it useful here: it tells us whether changing the threshold is
    genuinely improving the model's ranking ability, or merely trading
    precision for recall along a fixed curve.
    """
    churner_scores     = churn_probabilities[true_labels == 1]
    non_churner_scores = churn_probabilities[true_labels == 0]

    if len(churner_scores) == 0 or len(non_churner_scores) == 0:
        return float("nan")

    # Rank every score; ties share the average rank.
    all_scores = np.concatenate([churner_scores, non_churner_scores])
    order      = all_scores.argsort()
    ranks      = np.empty(len(all_scores), dtype=float)
    ranks[order] = np.arange(1, len(all_scores) + 1)

    # Average the ranks within each group of tied scores.
    unique_scores, inverse = np.unique(all_scores, return_inverse=True)
    for group_index in range(len(unique_scores)):
        tied = (inverse == group_index)
        if np.sum(tied) > 1:
            ranks[tied] = ranks[tied].mean()

    n_churners     = len(churner_scores)
    n_non_churners = len(non_churner_scores)
    sum_of_churner_ranks = ranks[:n_churners].sum()

    return (sum_of_churner_ranks - n_churners * (n_churners + 1) / 2) / (
        n_churners * n_non_churners
    )


# ---------------------------------------------------------------------------
# One row of the results table
# ---------------------------------------------------------------------------

def evaluate(system_name, true_labels, churn_probabilities, threshold):
    """
    Score one system and return a single row for the results table.

    The threshold is recorded alongside the metrics because a result is
    meaningless without knowing which cutoff produced it.
    """
    predictions = probabilities_to_predictions(churn_probabilities, threshold)
    counts      = count_confusion(true_labels, predictions)
    precision, recall, f1 = precision_recall_f1(true_labels, predictions)

    return {
        "System":     system_name,
        "Threshold":  round(float(threshold), 3),
        "Precision":  precision,
        "Recall":     recall,
        "F1":         f1,
        "AUC-ROC":    area_under_roc(true_labels, churn_probabilities),
        "TP":         counts["true_positives"],
        "FP":         counts["false_positives"],
        "FN":         counts["false_negatives"],
        "TN":         counts["true_negatives"],
        "Cost":       business_cost(true_labels, predictions),
    }


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def find_best_threshold(true_labels, churn_probabilities):
    """
    Sweep every sensible cutoff and return the one with the lowest cost.

    IMPORTANT - this must only ever be called on TRAINING data.
    Tuning a threshold on the test set and then reporting test results is
    data leakage, and it would invalidate every number in the report.

    Every baseline in this project gets this same treatment, so that the
    comparison against the annealing search stays fair in both directions.
    """
    candidate_thresholds = np.linspace(
        config.MIN_THRESHOLD, config.MAX_THRESHOLD, 181
    )

    best_threshold = config.DEFAULT_THRESHOLD
    best_cost      = float("inf")

    for threshold in candidate_thresholds:
        cost = business_cost_of_probabilities(
            true_labels, churn_probabilities, threshold
        )
        if cost < best_cost:
            best_cost      = cost
            best_threshold = threshold

    return float(best_threshold), float(best_cost)

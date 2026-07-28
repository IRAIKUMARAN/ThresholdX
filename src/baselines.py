"""
baselines.py
============
The systems we must beat, kept in their own file so nobody can accuse us of
quietly handicapping them.

This project claims that an AI search over ensemble weights is better than the
standard approach. That claim is only worth anything if the standard approach
is given a fair fight.

THE BUG THIS FILE FIXES
-----------------------
The original stacking baseline was built like this:

    meta = LogisticRegression(max_iter=1000, random_state=42)

with no class_weight, while ALL FOUR base models used
class_weight="balanced". Since only ~26% of customers churn, the unbalanced
meta-classifier learned to stay quiet: recall 0.297, business cost 1034.

The progress report then presented "cost reduced from 1034 to 675, a 35%
improvement" as the project's headline result. That 35% was measuring one
missing constructor argument, not the value of the search.

With the argument restored, this baseline becomes genuinely competitive, and
the comparison in the final report means something.

A SECOND FAIRNESS RULE
----------------------
Our proposed system is allowed to tune its decision threshold. So every
baseline here is allowed to tune its threshold too, on the same training data,
using the same function. Otherwise we would simply have flipped the unfairness
around to face the other way.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

import config
from src import cost as cost_module


def stacking_with_logistic_regression(oof_probabilities, y_train, test_probabilities):
    """
    The textbook stacking ensemble, and the baseline the proposal names.

    A logistic regression sits on top of the four base models and learns how to
    combine their probabilities. This is exactly what our annealing search
    replaces, so it is the comparison that matters most.

    It trains on out-of-fold predictions - the same clean inputs our search
    gets - so neither approach has an information advantage.
    """
    meta_classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",     # THE FIX - matches the base models
        random_state=config.RANDOM_SEED,
    )
    meta_classifier.fit(oof_probabilities, y_train)

    train_predictions = meta_classifier.predict_proba(oof_probabilities)[:, 1]
    test_predictions  = meta_classifier.predict_proba(test_probabilities)[:, 1]

    return train_predictions, test_predictions


def uniform_ensemble(oof_probabilities, test_probabilities):
    """
    Simply average the four models equally - no learning at all.

    This is the "did the search actually do anything?" control. If optimised
    weights cannot beat a plain average, the optimisation is not earning its
    place in the pipeline.
    """
    model_count    = oof_probabilities.shape[1]
    equal_weights  = np.ones(model_count) / model_count

    return (oof_probabilities @ equal_weights,
            test_probabilities @ equal_weights)


def strongest_single_model(oof_probabilities, y_train, model_names):
    """
    Find whichever individual classifier has the lowest training cost on its
    own, once its threshold is tuned.

    This is the most important and most uncomfortable baseline. If a four-model
    ensemble driven by simulated annealing cannot beat one plain classifier,
    then all the machinery is decoration. The original results table showed
    logistic regression alone at cost 679 versus the proposed system at 677 -
    which is why this baseline now appears explicitly rather than being left
    for a reader to notice.
    """
    best_name      = None
    best_column    = None
    best_threshold = config.DEFAULT_THRESHOLD
    best_cost      = float("inf")

    for column, name in enumerate(model_names):
        threshold, model_cost = cost_module.find_best_threshold(
            y_train, oof_probabilities[:, column]
        )
        if model_cost < best_cost:
            best_cost      = model_cost
            best_threshold = threshold
            best_name      = name
            best_column    = column

    return best_name, best_column, best_threshold, best_cost

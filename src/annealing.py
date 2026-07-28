"""
annealing.py
============
Simulated Annealing - the AI Search component of this project.

WHAT IT DOES
------------
We have four trained classifiers. Each one outputs a churn probability for
every customer. We want to blend them into a single better prediction, which
means choosing:

  * four weights  - how much to trust each classifier (must sum to 1)
  * one threshold - the cutoff that turns the blended probability into a
                    yes/no decision

There are infinitely many combinations, so we cannot try them all. Simulated
Annealing explores this space intelligently.

HOW IT WORKS
------------
Imagine standing blindfolded on hilly ground, trying to find the lowest point.
If you only ever step downhill you get stuck in the first small dip you find.
Simulated Annealing sometimes deliberately steps UPHILL so it can escape those
dips and find the real valley.

How often it accepts an uphill step is governed by the "temperature":

    hot  (start) -> accepts most uphill steps -> explores widely
    cold (end)   -> accepts almost none       -> settles into the best valley

The name comes from metallurgy: heat metal so its atoms move freely, then cool
it slowly so they settle into a strong, ordered structure.

THREE BUGS THAT WERE FIXED HERE
-------------------------------
1. The starting temperature was hardcoded to 1.0 while costs were in the
   thousands, so exp(-delta/T) underflowed to zero and an uphill step was
   NEVER accepted. It was greedy hill-climbing wearing the name "annealing".
   Fixed by calibrating the temperature against the measured cost scale.

2. The cooling rate was hardcoded to 0.995, which hit the stopping temperature
   after ~1,838 steps. The run therefore ended early and silently, even though
   the budget said 5,000. Fixed by deriving the cooling rate from the budget.

3. The threshold was frozen at 0.5 and only the weights were searched. Because
   the four classifiers are trained on the same data their probabilities are
   highly correlated, so reweighting them barely changed any decision - the
   search space was almost flat. Adding the threshold gives the search a
   dimension that actually moves the cost.
"""

import numpy as np

import config
from src import cost as cost_module


# ---------------------------------------------------------------------------
# A candidate solution
# ---------------------------------------------------------------------------

class Blend:
    """
    One point in the search space: how much we trust each model, plus the
    cutoff we use to make the final call.
    """

    def __init__(self, weights, threshold):
        self.weights   = np.asarray(weights, dtype=float)
        self.threshold = float(threshold)

    def predict_probabilities(self, probability_matrix):
        """
        Blend the four models' probabilities into one number per customer.

        probability_matrix has one row per customer and one column per model,
        so the matrix product gives the weighted average directly.
        """
        return probability_matrix @ self.weights

    def copy(self):
        return Blend(self.weights.copy(), self.threshold)

    def __repr__(self):
        weight_text = ", ".join(f"{w:.3f}" for w in self.weights)
        return f"Blend(weights=[{weight_text}], threshold={self.threshold:.3f})"


# ---------------------------------------------------------------------------
# The objective we are minimising
# ---------------------------------------------------------------------------

def evaluate_blend(blend, probability_matrix, true_labels):
    """
    The business cost of a candidate blend. Lower is better.

    This is the ONLY thing the search is allowed to look at. In particular it
    never sees the test set - the search runs entirely on out-of-fold training
    predictions.
    """
    blended = blend.predict_probabilities(probability_matrix)
    return cost_module.business_cost_of_probabilities(
        true_labels, blended, blend.threshold
    )


# ---------------------------------------------------------------------------
# Generating a neighbouring solution
# ---------------------------------------------------------------------------

def random_neighbour(blend, random_generator, search_threshold):
    """
    Take a small random step from the current solution.

    The weights are nudged by Gaussian noise, then clipped positive and
    renormalised so they still sum to 1 - that keeps us on the valid simplex
    of "trust proportions".

    The threshold is nudged too, but only when search_threshold is True. That
    flag is what separates the original 4-dimensional search from the fixed
    5-dimensional one, so both can be run and compared.
    """
    nudged_weights = blend.weights + random_generator.normal(
        0.0, config.SA_WEIGHT_STEP_SIZE, size=len(blend.weights)
    )
    # A weight can never be negative - "negative trust" is meaningless.
    nudged_weights = np.clip(nudged_weights, 1e-6, None)
    nudged_weights = nudged_weights / nudged_weights.sum()

    if search_threshold:
        nudged_threshold = blend.threshold + random_generator.normal(
            0.0, config.SA_THRESHOLD_STEP_SIZE
        )
        nudged_threshold = float(np.clip(
            nudged_threshold, config.MIN_THRESHOLD, config.MAX_THRESHOLD
        ))
    else:
        nudged_threshold = blend.threshold

    return Blend(nudged_weights, nudged_threshold)


# ---------------------------------------------------------------------------
# Temperature calibration  (this is the fix for bug 1)
# ---------------------------------------------------------------------------

def calibrate_temperatures(start_blend, probability_matrix, true_labels,
                           random_generator, search_threshold):
    """
    Work out sensible start and end temperatures by measuring the data.

    The Metropolis rule accepts an uphill move with probability

        exp(-cost_increase / temperature)

    so "temperature" only means something relative to the size of a typical
    cost increase. Rearranging for the temperature that yields a desired
    acceptance rate:

        temperature = -typical_cost_increase / ln(desired_acceptance_rate)

    We estimate typical_cost_increase by taking random steps and averaging the
    ones that made things worse. This removes the magic number entirely and
    makes the schedule adapt to any dataset or cost setting.
    """
    current_cost = evaluate_blend(start_blend, probability_matrix, true_labels)

    cost_increases = []
    for _ in range(config.SA_CALIBRATION_SAMPLES):
        candidate = random_neighbour(start_blend, random_generator, search_threshold)
        candidate_cost = evaluate_blend(candidate, probability_matrix, true_labels)
        if candidate_cost > current_cost:
            cost_increases.append(candidate_cost - current_cost)

    # Degenerate case: nothing we tried made it worse. Fall back to a small
    # positive temperature so the search still runs.
    if not cost_increases:
        return 1.0, 0.01

    typical_increase = float(np.mean(cost_increases))

    start_temperature = -typical_increase / np.log(config.SA_ACCEPT_RATE_AT_START)
    end_temperature   = -typical_increase / np.log(config.SA_ACCEPT_RATE_AT_END)

    return start_temperature, end_temperature


# ---------------------------------------------------------------------------
# The search itself
# ---------------------------------------------------------------------------

def simulated_annealing(probability_matrix, true_labels,
                        search_threshold=True,
                        max_iterations=None,
                        random_seed=None,
                        verbose=True):
    """
    Find the blend of models (and cutoff) with the lowest business cost.

    Parameters
    ----------
    probability_matrix : array, one row per customer, one column per model.
                         These MUST be out-of-fold predictions, so the search
                         never sees a model's opinion about data it trained on.
    true_labels        : the real churn outcomes for those same customers.
    search_threshold   : True  -> search weights AND threshold  (5 dimensions)
                         False -> search weights only, cutoff pinned at 0.5
                                  (4 dimensions - the original design)

    Returns a dict holding the best blend found, its cost, and the full history
    so the convergence can be plotted.
    """
    if max_iterations is None:
        max_iterations = config.SA_MAX_ITERATIONS
    if random_seed is None:
        random_seed = config.RANDOM_SEED

    random_generator = np.random.default_rng(random_seed)
    model_count      = probability_matrix.shape[1]

    # Start from equal trust in every model and the naive 0.5 cutoff, so any
    # improvement we report is improvement over the obvious default.
    current = Blend(
        weights=np.ones(model_count) / model_count,
        threshold=config.DEFAULT_THRESHOLD,
    )
    current_cost = evaluate_blend(current, probability_matrix, true_labels)

    best      = current.copy()
    best_cost = current_cost

    # Set the temperature schedule from the data rather than by guessing.
    start_temperature, end_temperature = calibrate_temperatures(
        current, probability_matrix, true_labels,
        random_generator, search_threshold
    )

    # Geometric cooling: T <- T * cooling_rate each step. We choose the rate so
    # the temperature lands exactly on end_temperature at the final iteration,
    # which guarantees the whole iteration budget is actually used.
    cooling_rate = (end_temperature / start_temperature) ** (1.0 / max_iterations)

    if verbose:
        dimensions = "5-D (weights + threshold)" if search_threshold else "4-D (weights only)"
        print(f"  Search space      : {dimensions}")
        print(f"  Starting cost     : {current_cost:.0f}")
        print(f"  Temperature       : {start_temperature:.2f} -> {end_temperature:.2f} "
              f"(cooling rate {cooling_rate:.5f})")

    temperature      = start_temperature
    accepted_moves   = 0
    uphill_accepted  = 0
    history          = []

    for iteration in range(max_iterations):
        candidate      = random_neighbour(current, random_generator, search_threshold)
        candidate_cost = evaluate_blend(candidate, probability_matrix, true_labels)

        cost_change = candidate_cost - current_cost

        # ---- Metropolis acceptance rule -----------------------------------
        # Better solution      -> always take it.
        # Worse solution       -> take it with probability exp(-increase / T),
        #                         which is high while hot and tiny once cold.
        if cost_change < 0:
            accept = True
        else:
            accept = random_generator.random() < np.exp(-cost_change / temperature)
            if accept:
                uphill_accepted += 1

        if accept:
            current      = candidate
            current_cost = candidate_cost
            accepted_moves += 1

        # Remember the best solution ever seen, separately from where the
        # search happens to be standing right now.
        if current_cost < best_cost:
            best_cost = current_cost
            best      = current.copy()

        history.append({
            "iteration":   iteration,
            "temperature": temperature,
            "current_cost": current_cost,
            "best_cost":    best_cost,
        })

        temperature *= cooling_rate

    acceptance_rate        = accepted_moves / max_iterations
    uphill_acceptance_rate = uphill_accepted / max_iterations

    if verbose:
        print(f"  Final cost        : {best_cost:.0f}")
        print(f"  Best threshold    : {best.threshold:.3f}")
        print(f"  Acceptance rate   : {acceptance_rate:.1%} "
              f"(of which uphill escapes: {uphill_acceptance_rate:.1%})")

    return {
        "best_blend":             best,
        "best_cost":              best_cost,
        "history":                history,
        "acceptance_rate":        acceptance_rate,
        "uphill_acceptance_rate": uphill_acceptance_rate,
        "start_temperature":      start_temperature,
        "end_temperature":        end_temperature,
        "iterations":             max_iterations,
    }

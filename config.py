"""
config.py
=========
Every number you can tune in this project lives here, and nowhere else.

Why one file? So that when someone asks "where did 5 come from?" or
"what threshold did you use?", there is exactly one place to look.
No magic numbers buried inside the algorithms.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Where things live
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
DATA_FILE    = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"
RESULTS_DIR  = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------------
# The business problem
# ---------------------------------------------------------------------------
# A "false negative" is a customer who churned but we predicted they would stay.
# We lose them completely, so we lose all their future revenue.
#
# A "false positive" is a loyal customer we wrongly flagged as churning.
# We waste a retention offer on them - annoying, but cheap.
#
# Losing a customer is roughly 5x worse than wasting a discount, so:

COST_OF_MISSING_A_CHURNER = 5.0   # penalty per false negative
COST_OF_FALSE_ALARM       = 1.0   # penalty per false positive

# Total business cost = 5 * (false negatives) + 1 * (false positives)
# Lower is better. This is the number the whole project tries to minimise.


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

TEST_SET_FRACTION = 0.20   # hold out 20% of customers for final testing
RANDOM_SEED       = 42     # fixed so results reproduce exactly
CV_FOLD_COUNT     = 5      # folds for out-of-fold base model training


# ---------------------------------------------------------------------------
# Simulated Annealing search
# ---------------------------------------------------------------------------

SA_MAX_ITERATIONS = 5000    # how many candidate solutions to try

# Temperature controls how willing the search is to accept a WORSE solution
# in order to escape a local optimum.
#
# The old code hardcoded the starting temperature to 1.0 while the cost values
# were in the thousands. That made exp(delta / T) underflow to zero, so a worse
# move was never accepted and the "annealing" degenerated into plain greedy
# hill-climbing (measured acceptance rate: 1.2%).
#
# Instead of guessing a number, we now measure the typical cost difference
# between neighbouring solutions and set the temperature from that. The two
# values below say what fraction of worse moves we want to accept at the very
# start of the search (exploring) and at the very end (settling down).

SA_ACCEPT_RATE_AT_START = 0.80   # start hot: accept 80% of worse moves
SA_ACCEPT_RATE_AT_END   = 0.01   # end cold: accept 1% of worse moves
SA_CALIBRATION_SAMPLES  = 200    # random probes used to measure the cost scale

# How far a single step can move the weights. Too small and the search crawls,
# too large and it bounces around randomly without settling.
SA_WEIGHT_STEP_SIZE    = 0.08
SA_THRESHOLD_STEP_SIZE = 0.03

# ---------------------------------------------------------------------------
# Guarding the search against overfitting
# ---------------------------------------------------------------------------
# The search tunes five numbers. Anything with five free parameters can find
# patterns that exist only in the data it was tuned on and vanish on new
# customers. That is overfitting, and a search is especially prone to it
# because its whole job is to hunt for whatever scores best.
#
# The guard: hold back a slice of the training data that the search cannot see
# while it explores. The search still EXPLORES using the larger portion, but
# the solution we finally KEEP is whichever scored best on the held-back
# slice. A solution that only works on the data it was fitted to will lose
# here, so it never gets chosen.
#
# This is separate from, and additional to, the final test set - which stays
# untouched until the very end regardless.

# MEASURED RESULT: this guard is switched OFF by default, because we tested it
# and it did not work.
#
# On a controlled benchmark it made performance on unseen data WORSE - 0.3124
# against 0.3008 cost per customer. Two reasons, both real:
#
#   1. Selecting the best of ~5,000 candidate solutions against a held-back
#      slice means you start fitting the slice itself. The guard against
#      overfitting becomes a new way to overfit.
#   2. Shrinking the explored data from 100% to 75% makes the objective
#      noisier, so the search has a worse signal to follow.
#
# The mechanism is implemented, tested and available. Turn it on to reproduce
# the comparison, and see Experiment 5 in analyse_result.py, which settles the
# question on the real dataset rather than on a benchmark.

SA_VALIDATION_FRACTION = 0.25    # share of training data held back for selection
SA_USE_VALIDATION      = False   # measured as unhelpful - see note above

# The k-fold alternative. Instead of holding back one slice, run the entire
# search k times on k different slices and average the answers. Answers that
# only work on one particular slice disagree with each other and average away.
#
# This attacks the same overfitting problem as the guard above, but without its
# weakness - there is no single small slice to accidentally fit.
SA_CV_FOLDS = 5


# ---------------------------------------------------------------------------
# Decision threshold
# ---------------------------------------------------------------------------
# Every model outputs a probability between 0 and 1. We need a cutoff to turn
# that into a yes/no answer.
#
# The obvious cutoff is 0.5, but that is only correct when both mistakes cost
# the same. Ours do not - missing a churner costs 5x a false alarm - so the
# correct cutoff is well below 0.5. Letting the search find it is the point.

DEFAULT_THRESHOLD = 0.50    # the naive cutoff, kept for comparison
MIN_THRESHOLD     = 0.05    # search bounds: never go fully "predict everyone"
MAX_THRESHOLD     = 0.95


# ---------------------------------------------------------------------------
# Segment-specific thresholds
# ---------------------------------------------------------------------------
# Instead of one global cutoff for every customer, tune a separate cutoff per
# value of this raw (unencoded) column. Contract type is the strongest single
# churn signal in the dataset and splits customers into three groups with
# very different baseline churn rates, which is why it was the first thing
# tried here. Change this to experiment with other segment columns
# (e.g. "InternetService", "PaymentMethod") without touching any other code.
#
# MEASURED RESULT with "Contract": this did NOT help, and made things
# slightly worse. Across 5 train/test splits, segmenting the SA 5-D
# ensemble's threshold by Contract cost +9.2 on average versus its own
# global threshold, and did not close the gap to Logistic Regression either
# (mean +14.0, 95% CI [-12.0, +40.0] - includes zero, so not even a
# significant loss, just no measured benefit).
#
# The likely reason: Contract type is already one of the 30 features all
# four base models see and weight heavily (it's the single strongest churn
# signal). The ensemble's probability output already encodes "this customer
# is month-to-month, therefore high risk" - so tuning a second, separate
# threshold on the same variable doesn't add new information, it just
# splits the training data into three smaller, noisier slices to tune
# against. Same failure mode as the validation guard above: less data per
# decision, no new signal, more noise.
#
# Kept in the code (not deleted) because it's a legitimate thing to have
# tried, and a different segment column - one that carries information the
# base models DON'T already use heavily - might behave differently. Contract
# was simply the wrong candidate.

SEGMENT_COLUMN = "Contract"


# The BBN only gets consulted when the ensemble is genuinely unsure, i.e. when
# its probability lands inside this band around the decision threshold.

BBN_UNCERTAINTY_BAND = 0.10   # +/- around the threshold

# Laplace smoothing added to every count when learning the probability tables,
# so that a combination never seen in training does not get probability zero.
BBN_SMOOTHING = 1.0

# Bin edges for turning continuous features into the discrete states a
# Bayesian network needs.
TENURE_BINS   = [0, 24, 48, 100]      # months: new / established / loyal
MONTHLY_BINS  = [0, 45, 75, 1000]     # dollars: cheap / mid / premium
TENURE_LABELS  = ["New", "Established", "Loyal"]
MONTHLY_LABELS = ["Cheap", "Mid", "Premium"]


# ---------------------------------------------------------------------------
# Robustness check
# ---------------------------------------------------------------------------
# One run proves nothing - a 2-unit cost difference could be luck. We repeat
# the whole search with different random seeds and report the spread.

ROBUSTNESS_SEEDS = [42, 7, 123, 2024, 999]

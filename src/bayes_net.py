"""
bayes_net.py
============
The Bayesian Belief Network - the Knowledge Representation component.

WHAT IT IS FOR
--------------
The ensemble is confident about most customers. For a minority it lands near
the decision cutoff and is essentially guessing. For those uncertain cases we
ask a second, completely different kind of model: one that reasons over
meaningful customer facts rather than blending numeric probabilities.

STRUCTURE
---------
                        Churned
              /      /      |      \\       \\
      tenure_group  contract  monthly  internet  tech_support ...

Each customer fact depends on whether the customer churned. Reading the arrows
backwards with Bayes' rule gives what we actually want:

    P(churn | facts)  proportional to  P(churn) * P(fact_1|churn) * P(fact_2|churn) * ...

Every one of those probabilities is LEARNED BY COUNTING the training data.

TWO BUGS THIS FILE FIXES
------------------------
1. The old code read the contract type from one-hot columns named
   "Contract_One year" / "Contract_Two year". But it was handed the RAW table,
   where the column is simply "Contract" holding text. The lookup silently
   failed and fell through to a default, so ALL 7,043 customers were recorded
   as "Month-to-month". Verified by counting. Contract type is the single
   strongest churn signal in this dataset, and it was a dead constant.

2. The file's docstring claimed the probability tables were "learned from
   training data via Maximum Likelihood Estimation". They were not. The two
   tables that mattered were typed in by hand:

       values=[[0.30, 0.50, 0.70, 0.10, 0.30, 0.50], ...]

   Only three marginal priors were counted from data, and inference conditioned
   on all of those anyway, so they cancelled out. The network's output did not
   depend on the training labels AT ALL - calling fit() changed nothing.

   A note on structure: the old network also had a hidden "ChurnRisk" node.
   A hidden node cannot be learned by counting - it needs Expectation
   Maximisation - which is precisely why those tables ended up hand-written.
   We use a structure with no hidden node, so every number is learned honestly
   from data and every number can be explained in the report.
"""

import numpy as np
import pandas as pd

import config


# The customer facts the network reasons over. Each must be categorical.
EVIDENCE_COLUMNS = [
    "tenure_group",
    "Contract",
    "monthly_group",
    "InternetService",
    "TechSupport",
    "PaymentMethod",
]


def describe_customers(raw_table):
    """
    Turn raw customer rows into the discrete facts the network uses.

    Continuous numbers get bucketed, because a Bayesian network reasons over
    a finite set of states rather than over real numbers.

    Note this reads "Contract" straight from the original text column, which
    is the fix for bug 1 above.
    """
    facts = pd.DataFrame(index=raw_table.index)

    # How long has this customer been with us?
    facts["tenure_group"] = pd.cut(
        raw_table["tenure"],
        bins=config.TENURE_BINS,
        labels=config.TENURE_LABELS,
        include_lowest=True,
    ).astype(str)

    # How much do they pay per month?
    facts["monthly_group"] = pd.cut(
        raw_table["MonthlyCharges"],
        bins=config.MONTHLY_BINS,
        labels=config.MONTHLY_LABELS,
        include_lowest=True,
    ).astype(str)

    # These are already categorical text in the source file - use them as-is.
    facts["Contract"]        = raw_table["Contract"].astype(str)
    facts["InternetService"] = raw_table["InternetService"].astype(str)
    facts["TechSupport"]     = raw_table["TechSupport"].astype(str)
    facts["PaymentMethod"]   = raw_table["PaymentMethod"].astype(str)

    return facts[EVIDENCE_COLUMNS]


class ChurnBayesNet:
    """
    A Bayesian network over customer facts, learned entirely by counting.

    Nothing in this class is hand-tuned. Every probability comes from the
    training data, which means calling fit() with different labels genuinely
    changes what the network believes.
    """

    def __init__(self, smoothing=None):
        # Laplace smoothing: add a small count to every combination so that a
        # fact never seen alongside churn does not get probability zero and
        # veto everything else.
        self.smoothing = config.BBN_SMOOTHING if smoothing is None else smoothing

        self.churn_prior = None            # P(Churned)
        self.fact_given_churn = {}         # P(fact = value | Churned = yes/no)
        self.known_values = {}

        # Calibration parameters - see _fit_calibration below.
        self.calibration_scale = 1.0
        self.calibration_shift = 0.0

    def fit(self, raw_table, labels):
        """
        Learn every probability in the network by counting training customers.

        This is Maximum Likelihood Estimation: the best estimate of
        P(contract = "Month-to-month" | churned) is simply the fraction of
        churned customers who were on a month-to-month contract.
        """
        facts  = describe_customers(raw_table)
        labels = np.asarray(labels)

        if len(facts) != len(labels):
            raise ValueError(
                f"Got {len(facts)} customers but {len(labels)} labels. "
                "These must describe the same people in the same order."
            )

        # ---- P(Churned) -------------------------------------------------
        self.churn_prior = float(np.mean(labels))

        churned_rows = (labels == 1)
        stayed_rows  = (labels == 0)

        # ---- P(fact | Churned) for every fact and every value -----------
        self.fact_given_churn = {}
        self.known_values = {}

        for column in EVIDENCE_COLUMNS:
            values = sorted(facts[column].unique())
            self.known_values[column] = values

            table = {}
            for value in values:
                matches = (facts[column] == value).values

                # Fraction of churners showing this value, and of non-churners.
                churn_count = np.sum(matches & churned_rows) + self.smoothing
                stay_count  = np.sum(matches & stayed_rows)  + self.smoothing

                churn_total = np.sum(churned_rows) + self.smoothing * len(values)
                stay_total  = np.sum(stayed_rows)  + self.smoothing * len(values)

                table[value] = {
                    "given_churned": churn_count / churn_total,
                    "given_stayed":  stay_count / stay_total,
                }

            self.fact_given_churn[column] = table

        print(f"  Learned P(churn) = {self.churn_prior:.3f} "
              f"and {len(EVIDENCE_COLUMNS)} conditional tables from "
              f"{len(labels):,} customers")

        self._fit_calibration(raw_table, labels)

        return self

    def _fit_calibration(self, raw_table, labels):
        """
        Correct the network's overconfidence.

        The structure assumes the customer facts are independent of each other
        once you know whether the customer churned. They are not - internet
        service, tech support and contract type overlap heavily. Multiplying
        their likelihoods therefore counts the same underlying evidence several
        times, and the resulting probabilities get pushed toward 0 and 1.

        Measured on the real dataset: the uncalibrated network predicted a
        32.8% churn rate where the truth was 27.2%.

        The standard remedy is Platt scaling - stretch and shift the log-odds
        by two numbers learned from the training data:

            calibrated_log_odds = scale * raw_log_odds + shift

        Two parameters cannot change which customers the network ranks as
        riskier than which - so this fixes the SCALE of the probabilities
        without touching what the network actually learned. That matters
        because these probabilities get compared against the same decision
        threshold as the ensemble's, so the two must mean the same thing.
        """
        raw_log_odds = self._log_odds(raw_table)
        labels = np.asarray(labels, dtype=float)

        scale, shift = 1.0, 0.0

        # Fit the two parameters by gradient descent on log loss.
        for _ in range(300):
            predicted = 1.0 / (1.0 + np.exp(-np.clip(scale * raw_log_odds + shift, -30, 30)))
            error = predicted - labels
            gradient_scale = np.mean(error * raw_log_odds)
            gradient_shift = np.mean(error)
            scale -= 0.5 * gradient_scale
            shift -= 0.5 * gradient_shift

        self.calibration_scale = float(scale)
        self.calibration_shift = float(shift)

        print(f"  Calibrated the network: log-odds scaled by "
              f"{self.calibration_scale:.3f}, shifted by {self.calibration_shift:+.3f}")

    def _log_odds(self, raw_table):
        """
        The raw, uncalibrated evidence for churn, expressed as log-odds:

            log P(churned | facts) - log P(stayed | facts)

        Positive means the facts point toward churn. Working in log space keeps
        things stable - multiplying six small probabilities together would
        underflow toward zero, whereas adding their logarithms does not.
        """
        if self.churn_prior is None:
            raise RuntimeError("Call fit() before predicting.")

        facts = describe_customers(raw_table)

        log_churn = np.full(len(facts), np.log(self.churn_prior))
        log_stay  = np.full(len(facts), np.log(1.0 - self.churn_prior))

        for column in EVIDENCE_COLUMNS:
            table  = self.fact_given_churn[column]
            values = facts[column].values

            # A value never seen in training falls back to "uninformative":
            # equally likely under either outcome, so it changes nothing.
            fallback = {"given_churned": 0.5, "given_stayed": 0.5}

            log_churn += np.array([
                np.log(table.get(value, fallback)["given_churned"]) for value in values
            ])
            log_stay += np.array([
                np.log(table.get(value, fallback)["given_stayed"]) for value in values
            ])

        return log_churn - log_stay

    def predict_probability(self, raw_table):
        """
        P(churned | this customer's facts), after calibration.

        Bayes' rule gives the raw evidence; the two calibration parameters
        learned during fit() put it on a scale that matches reality.
        """
        raw_log_odds = self._log_odds(raw_table)
        calibrated = self.calibration_scale * raw_log_odds + self.calibration_shift

        return 1.0 / (1.0 + np.exp(-np.clip(calibrated, -30, 30)))


def apply_to_uncertain_cases(ensemble_probabilities, bayes_probabilities,
                             threshold, band=None):
    """
    Replace the ensemble's opinion with the network's, but ONLY where the
    ensemble is genuinely undecided - that is, where its probability sits
    within `band` of the decision cutoff.

    Confident predictions are left completely untouched.

    Note the band is centred on the ACTUAL threshold being used, not on a
    hardcoded 0.5. When the search lowers the cutoff to catch more churners,
    the "uncertain zone" has to move with it.
    """
    if band is None:
        band = config.BBN_UNCERTAINTY_BAND

    is_uncertain = (
        (ensemble_probabilities >= threshold - band) &
        (ensemble_probabilities <= threshold + band)
    )

    adjusted = ensemble_probabilities.copy()
    adjusted[is_uncertain] = bayes_probabilities[is_uncertain]

    count = int(np.sum(is_uncertain))
    print(f"  Ensemble was uncertain about {count:,} of "
          f"{len(ensemble_probabilities):,} customers "
          f"({count / len(ensemble_probabilities):.1%}); "
          f"deferred those to the Bayesian network")

    return adjusted

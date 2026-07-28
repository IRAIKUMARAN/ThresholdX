# ThresholdX

**Cost-Aware Ensemble Optimisation for Customer Churn Prediction using Simulated Annealing**

ECE 569A — Selected Topics in Computer Engineering: Artificial Intelligence
University of Victoria · Grad Group 21

| Member | Student ID |
|---|---|
| Anand Anto | V01098042 |
| Nalluraj Babu | V01096399 |
| Irai Kumaran Sivanesan | V01092990 |

---

## Overview

Customer churn is an asymmetric prediction problem. Losing a customer costs far
more than wasting a retention offer on someone who was never going to leave, so
a model optimised for plain accuracy is solving the wrong problem.

This project replaces the usual logistic-regression meta-classifier in a
stacking ensemble with **Simulated Annealing**, a heuristic AI search. The
search directly minimises a business cost

```
cost  =  5 × (missed churners)  +  1 × (false alarms)
```

by choosing how much to trust each of four base classifiers **and** where to
place the decision threshold. A **Bayesian Belief Network**, learned from the
data by maximum likelihood, then handles the customers the ensemble is least
certain about.

The project covers the three pillars of the course:

| Pillar | Component |
|---|---|
| Machine Learning | Four base classifiers with out-of-fold training |
| AI Search | Simulated Annealing over the weight simplex + threshold |
| Knowledge Representation | Bayesian Belief Network over customer facts |

---

## Quick start

```bash
git clone https://github.com/IRAIKUMARAN/ThresholdX.git
cd ThresholdX
python3 -m pip install -r requirements.txt

python3 tests/verify_dataset.py    # confirm the dataset is authentic
python3 tests/verify_logic.py      # 36 correctness checks
python3 run_pipeline.py            # full pipeline
```

Optional:

```bash
python3 run_pipeline.py --quick        # fewer iterations, fast check
python3 run_pipeline.py --seeds        # repeat search across 5 random seeds
python3 analyse_result.py              # diagnostic experiments
python3 analyse_result.py --splits 5   # repeat across 5 train/test splits
```

Results are written to `results/` — five figures, `results_table.csv` and
`run_record.json`.

Requires Python 3.9+. On Windows use `python` in place of `python3`.

---

## Repository structure

```
ThresholdX/
├── config.py              Every tunable parameter, in one place
├── run_pipeline.py        End-to-end pipeline (6 stages)
├── analyse_result.py      Diagnostic experiments explaining the result
├── requirements.txt
│
├── src/
│   ├── data_prep.py       Loading, cleaning, encoding, splitting
│   ├── base_models.py     Four classifiers + out-of-fold training
│   ├── annealing.py       Simulated Annealing search
│   ├── bayes_net.py       Bayesian Belief Network
│   ├── baselines.py       Comparison systems
│   ├── cost.py            Business cost function and metrics
│   └── plots.py           Figure generation
│
├── tests/
│   ├── verify_dataset.py  Confirms the dataset is the genuine IBM release
│   └── verify_logic.py    36 checks over the core algorithms
│
├── data/                  Dataset (see data/README.md)
└── results/               Generated output (regenerated on each run)
```

Two rules hold throughout: nothing in `src/` writes files, and nothing in
`plots.py` computes a number. Changing how a figure looks can never change a
result.

---

## Method

**1 — Data.** IBM Watson Telco Customer Churn: 7,043 customers, 26.54% churn
rate, 20 features. The 11 blank `TotalCharges` entries all belong to tenure-0
customers who had not yet been billed, so they are filled with 0 rather than
imputed. Numeric features are standardised using statistics fitted on the
training split only.

**2 — Base models.** Logistic Regression, Decision Tree, Random Forest and SVM,
all with `class_weight="balanced"`. Out-of-fold predictions are generated with
5-fold stratified cross-validation, so the search only ever sees each model's
opinion about customers it did not train on.

**3 — Search.** Simulated Annealing over a 5-dimensional space: four ensemble
weights constrained to the simplex, plus the decision threshold. The
temperature schedule is calibrated by measuring the actual spread of
neighbouring solution costs, rather than being hardcoded.

**4 — Bayesian network.** A network over six discretised customer facts —
tenure band, contract type, monthly charge band, internet service, tech support
and payment method. All conditional probability tables are learned by counting
with Laplace smoothing, then calibrated with Platt scaling to correct the
overconfidence that the naive independence assumption introduces.

**5 — Evaluation.** Every system, including every baseline, has its decision
threshold tuned on training data and applied unchanged to the test set. The
test set is touched exactly once, to report final numbers.

---

## Results

Business cost on the held-out test set (1,409 customers). Lower is better.

| System | Threshold | Recall | F1 | AUC | Cost |
|---|---|---|---|---|---|
| **Logistic Regression alone** | 0.320 | 0.920 | 0.592 | 0.842 | **594** |
| **SA Ensemble 5-D (proposed)** | 0.370 | 0.880 | 0.603 | 0.845 | **613** |
| Random Forest | 0.325 | 0.882 | 0.598 | 0.840 | 619 |
| Uniform Ensemble | 0.285 | 0.882 | 0.596 | 0.846 | 623 |
| LR Stacking (baseline) | 0.365 | 0.864 | 0.602 | 0.846 | 631 |
| Decision Tree | 0.365 | 0.885 | 0.587 | 0.826 | 637 |
| SVM | 0.095 | 0.888 | 0.580 | 0.825 | 648 |
| SA + Bayesian Network | 0.370 | 0.818 | 0.612 | 0.832 | 660 |
| SA Ensemble 4-D (weights only) | 0.500 | 0.791 | 0.624 | 0.844 | 669 |

Learned weights — Logistic Regression 0.708, Random Forest 0.252,
Decision Tree 0.041, SVM 0.000.

Across five random seeds the proposed system scores **615.4 ± 2.8**
(range 612–620).

### Discussion

**Searching the threshold matters.** Extending the search from four dimensions
to five improved the cost by 56 points (669 → 613, 8.4%) and beat the stacking
baseline by 18 points (2.9%). The optimal threshold is **0.370**, far below the
conventional 0.5 — the direct consequence of a 5:1 cost asymmetry. Holding the
threshold fixed at 0.5, as a conventional pipeline does, leaves most of the
available gain unclaimed.

**Ensemble weighting does not.** A single logistic regression with a tuned
threshold costs **594**, nineteen points better than the full four-model
ensemble. That margin sits well outside the ±2.8 seed-to-seed spread, so it is
a real difference rather than run-to-run noise. The search converges toward
this conclusion itself: it assigns 0.708 of the weight to logistic regression
and drives the SVM to exactly zero.

**The Bayesian network does not help either.** Deferring uncertain cases to the
network raises the cost from 613 to 660. The network is well calibrated and
achieves reasonable AUC on its own, but it is simply weaker than the ensemble
on the cases where the ensemble is unsure.

We report these as findings rather than omitting them. `run_pipeline.py`
compares the proposed system against the strongest single classifier on every
run and states plainly when it loses; `analyse_result.py` investigates why,
decomposing the improvement into threshold and weight contributions, measuring
how correlated the base classifiers are, and testing whether the additional
free parameters overfit.

---

## Reproducibility

Every result is deterministic given a fixed seed (`config.RANDOM_SEED = 42`).

- `tests/verify_dataset.py` — eight checks confirming the CSV is the genuine
  IBM release, including customer-ID format, churn rate and the known
  eleven-row `TotalCharges` gap. An earlier revision of this project ran on a
  file that resembled the IBM dataset but was not it; this script exists so
  that can never pass unnoticed again.
- `tests/verify_logic.py` — 36 checks covering the cost function, the metrics
  (AUC is validated against a brute-force pairwise computation), threshold
  tuning, the annealing schedule and acceptance behaviour, and the Bayesian
  network's learning and calibration.

---

## References

Russell, S., & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach*
(4th ed.). Pearson.

Platt, J. (1999). Probabilistic outputs for support vector machines and
comparisons to regularized likelihood methods. *Advances in Large Margin
Classifiers*.

IBM. *Telco Customer Churn Dataset.*
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

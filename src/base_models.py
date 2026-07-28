"""
base_models.py
==============
The four base classifiers, and the out-of-fold training that feeds the search.

WHY OUT-OF-FOLD?
----------------
The annealing search decides how much to trust each classifier. To do that
honestly it must see how well each one does on customers it has NOT already
memorised. If we trained a model and then asked it about the same customers,
an overfitted model would look brilliant and the search would over-trust it.

So we use 5-fold cross-validation:

  * split the training customers into 5 groups
  * hold out group 1, train on groups 2-5, predict group 1
  * repeat for each group

Every customer ends up with a prediction from a model that never saw them.
That clean matrix of predictions is what the search optimises against.

Separately, each model is retrained on the full training set. Those are the
models used on the held-out test set at the very end.
"""

import warnings

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

import config
from src import cost as cost_module

# scikit-learn 1.9 deprecated SVC(probability=True) in favour of wrapping the
# classifier in CalibratedClassifierCV. We keep the original form because the
# proposal specifies a plain SVM and switching would change the probabilities
# mid-project. The warning is silenced here, with this note, rather than left
# to print five times per run and bury the actual output.
warnings.filterwarnings(
    "ignore", message=".*probability.*parameter was deprecated.*"
)


def build_models():
    """
    The four classifiers named in the project proposal.

    Every one uses class_weight="balanced". Only about 26% of customers churn,
    so without this the models would learn that predicting "nobody churns" is
    a comfortable 74% correct - and useless. Balancing tells each model to
    treat the rare churn class as equally important.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,                       # shallow, to resist overfitting
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            n_jobs=-1,
            random_state=config.RANDOM_SEED,
        ),
        "SVM": SVC(
            kernel="rbf",
            probability=True,                  # needed: we want probabilities
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
        ),
    }


def train_out_of_fold(X_train, y_train, quiet=False):
    """
    Build the out-of-fold probability matrix and the final full-data models.

    Returns
    -------
    oof_probabilities : array (n_training_customers, 4)
        Column j holds model j's churn probability for each customer,
        always produced by a copy of the model that did not train on them.
    trained_models : dict of name -> model fitted on ALL training data
    model_names    : list of names, in the same order as the columns
    """
    models      = build_models()
    model_names = list(models.keys())

    oof_probabilities = np.zeros((len(X_train), len(models)))

    folds = StratifiedKFold(
        n_splits=config.CV_FOLD_COUNT,
        shuffle=True,
        random_state=config.RANDOM_SEED,
    )

    if not quiet:
        print(f"  Running {config.CV_FOLD_COUNT}-fold out-of-fold training "
              f"on {len(X_train):,} customers")

    for fold_number, (fit_rows, holdout_rows) in enumerate(folds.split(X_train, y_train), start=1):
        for column, (name, template) in enumerate(models.items()):
            # clone() gives a fresh unfitted copy with the same settings, so
            # folds never contaminate each other.
            model = clone(template)
            model.fit(X_train[fit_rows], y_train[fit_rows])
            oof_probabilities[holdout_rows, column] = \
                model.predict_proba(X_train[holdout_rows])[:, 1]

        if not quiet:
            print(f"    fold {fold_number}/{config.CV_FOLD_COUNT} complete")

    # Retrain each model on the complete training set for use on the test set.
    if not quiet:
        print("  Retraining each model on the full training set")
    trained_models = {}
    for name, template in models.items():
        model = clone(template)
        model.fit(X_train, y_train)
        trained_models[name] = model

    # Report how each model does on its own, at the naive 0.5 cutoff.
    if not quiet:
        print()
        print(f"    {'Model':<22}{'F1':>8}{'Recall':>9}{'Cost':>9}")
        print("    " + "-" * 48)
        for column, name in enumerate(model_names):
            probabilities = oof_probabilities[:, column]
            predictions = cost_module.probabilities_to_predictions(
                probabilities, config.DEFAULT_THRESHOLD
            )
            _, recall, f1 = cost_module.precision_recall_f1(y_train, predictions)
            model_cost = cost_module.business_cost(y_train, predictions)
            print(f"    {name:<22}{f1:>8.3f}{recall:>9.3f}{model_cost:>9.0f}")

    return oof_probabilities, trained_models, model_names


def predict_test_probabilities(trained_models, X_test):
    """
    Ask every fully-trained model for its churn probability on the test set.
    Column order matches the out-of-fold matrix, which is what lets the
    learned weights be applied unchanged.
    """
    return np.column_stack([
        model.predict_proba(X_test)[:, 1]
        for model in trained_models.values()
    ])

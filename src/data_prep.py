"""
data_prep.py
============
Loads the IBM Telco churn dataset and prepares it for the rest of the pipeline.

Two different views of the same customers are produced:

  1. An ENCODED, SCALED numeric matrix  - what the four classifiers need.
  2. The ORIGINAL, human-readable table - what the Bayesian network needs,
     because a Bayesian network reasons over meaningful categories like
     "Month-to-month contract", not over scaled floating-point numbers.

THE BUG THIS FILE FIXES
-----------------------
The old code split these two views SEPARATELY, then paired the shuffled
customer rows against labels sliced in the original file order:

    X_raw_train, X_raw_test = train_test_split(...)      # shuffled
    bbn.fit(X_raw_train, y_raw[:len(X_raw_train)], ...)  # NOT shuffled

Every customer was therefore matched with a stranger's churn outcome, and the
Bayesian network spent the whole project learning from noise.

The fix is structural rather than a patch: we split the customer INDICES once,
then use that single set of indices for every view of the data. It is now
impossible for the two views to disagree.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config


# Columns that are simple yes/no (or male/female) answers.
YES_NO_COLUMNS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]

# Columns with three or more categories, which become one-hot indicator columns.
CATEGORY_COLUMNS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]

# Genuinely numeric columns, which get standardised.
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_raw_table(csv_path=None, quiet=False):
    """
    Read the CSV and repair the one known data-quality problem.

    11 customers have a blank TotalCharges. They are all customers with
    tenure 0 - they signed up but had not yet been billed - so the blank
    genuinely means "nothing charged yet" rather than "value missing".
    Zero is therefore the honest fill value.

    (The original code imputed the median here, which quietly told the model
    that these brand-new customers had already paid an average lifetime bill.)
    """
    if csv_path is None:
        csv_path = config.DATA_FILE

    table = pd.read_csv(csv_path)

    # Blank strings become real missing values, then become 0.
    table["TotalCharges"] = pd.to_numeric(
        table["TotalCharges"].replace(" ", np.nan), errors="coerce"
    )
    blank_count = int(table["TotalCharges"].isna().sum())
    table["TotalCharges"] = table["TotalCharges"].fillna(0.0)

    # The label: did this customer leave?
    table["Churned"] = (table["Churn"] == "Yes").astype(int)

    if not quiet:
        print(f"  Loaded {len(table):,} customers, "
              f"{table['Churned'].mean():.1%} of them churned")
        print(f"  Repaired {blank_count} blank TotalCharges entries "
              f"(tenure-0 customers)")

    return table


def encode_features(table):
    """
    Turn the raw table into a purely numeric matrix the classifiers can use.
    Returns the matrix and the list of column names describing it.
    """
    features = table.drop(columns=["customerID", "Churn", "Churned"])

    # gender is the one binary column that is not Yes/No.
    features["gender"] = (features["gender"] == "Male").astype(int)

    for column in YES_NO_COLUMNS:
        features[column] = (features[column] == "Yes").astype(int)

    # One-hot encode the multi-category columns. drop_first avoids giving the
    # model a perfectly redundant column.
    features = pd.get_dummies(features, columns=CATEGORY_COLUMNS, drop_first=True)

    feature_names  = list(features.columns)
    feature_matrix = features.values.astype(float)

    return feature_matrix, feature_names


def prepare_data(csv_path=None, split_seed=None, quiet=False):
    """
    Produce every view of the data the pipeline needs, all split consistently.

    Returns a dict containing, for both train and test:
      * the scaled numeric matrix   (for the classifiers)
      * the original readable rows  (for the Bayesian network)
      * the churn labels

    All three are indexed by the same customers in the same order.

    split_seed lets an experiment re-split the same customers a different way.
    It exists so analyse_result.py can repeat the whole comparison across
    several train/test splits - a conclusion that only holds for one particular
    split is not a conclusion.
    """
    if split_seed is None:
        split_seed = config.RANDOM_SEED

    table = load_raw_table(csv_path, quiet=quiet)

    feature_matrix, feature_names = encode_features(table)
    labels = table["Churned"].values

    # ---- Split ONCE, by index. This is the fix. -------------------------
    all_row_positions = np.arange(len(table))
    train_positions, test_positions = train_test_split(
        all_row_positions,
        test_size=config.TEST_SET_FRACTION,
        stratify=labels,                 # keep the churn rate equal in both halves
        random_state=split_seed,
    )

    # ---- Scale numeric columns, fitting on TRAINING DATA ONLY -----------
    # Fitting the scaler on all the data first would leak information about
    # the test set into training.
    numeric_positions = [feature_names.index(c) for c in NUMERIC_COLUMNS]

    scaler = StandardScaler()
    scaler.fit(feature_matrix[np.ix_(train_positions, numeric_positions)])

    scaled_matrix = feature_matrix.copy()
    scaled_matrix[:, numeric_positions] = scaler.transform(
        feature_matrix[:, numeric_positions]
    )

    # ---- Every view, sliced with the SAME positions ---------------------
    data = {
        "X_train": scaled_matrix[train_positions],
        "X_test":  scaled_matrix[test_positions],
        "y_train": labels[train_positions],
        "y_test":  labels[test_positions],

        # Readable rows for the Bayesian network, same customers same order.
        "raw_train": table.iloc[train_positions].reset_index(drop=True),
        "raw_test":  table.iloc[test_positions].reset_index(drop=True),

        "feature_names": feature_names,
        "scaler": scaler,
    }

    if not quiet:
        print(f"  Training set: {len(data['y_train']):,} customers "
              f"({data['y_train'].mean():.1%} churn)")
        print(f"  Test set:     {len(data['y_test']):,} customers "
              f"({data['y_test'].mean():.1%} churn)")
        print(f"  Features after encoding: {len(feature_names)}")

    # A cheap assertion that would have caught the original bug immediately.
    assert len(data["raw_train"]) == len(data["y_train"]), \
        "Raw rows and labels are different lengths - the split is inconsistent."
    assert np.array_equal(data["raw_train"]["Churned"].values, data["y_train"]), \
        "Raw rows and labels disagree - customers are paired with wrong outcomes."

    return data

"""
verify_dataset.py
=================
Confirms that the CSV in data/ really is the IBM Watson Telco Customer Churn
dataset, and not a lookalike.

Run it with:      python tests/verify_dataset.py

WHY THIS EXISTS
---------------
The file originally shipped with this project was a synthetic imitation. It had
the right column names, the right number of rows, and roughly the right
marginal distributions, so nothing obvious looked wrong - but the churn rate,
the customer ID format, and the billing arithmetic all gave it away.

The project proposal and progress report both state that the IBM dataset was
used. This script makes that claim checkable in one command, so it can never
quietly stop being true again.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import re

import numpy as np
import pandas as pd

import config


# What the genuine IBM Watson Telco Customer Churn dataset contains.
EXPECTED_ROWS            = 7043
EXPECTED_CHURNERS        = 1869
EXPECTED_CHURN_RATE      = 0.2654
EXPECTED_BLANK_CHARGES   = 11
EXPECTED_MAX_TENURE      = 72
EXPECTED_ID_PATTERN      = r"\d{4}-[A-Z]{5}"     # e.g. 7590-VHVEG

problems = []


def check(description, condition, found):
    marker = "OK  " if condition else "FAIL"
    print(f"  [{marker}] {description}")
    if not condition:
        print(f"         found: {found}")
        problems.append(description)


print()
print("Verifying the dataset in data/")
print("=" * 70)

if not config.DATA_FILE.exists():
    print()
    print(f"  No file at {config.DATA_FILE}")
    print()
    print("  Download the real dataset from either:")
    print("    https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
          "/master/data/Telco-Customer-Churn.csv")
    print("    https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
    print()
    print("  and save it as data/Telco-Customer-Churn.csv")
    print()
    sys.exit(1)

customers = pd.read_csv(config.DATA_FILE)

# ---- Shape ---------------------------------------------------------------
check(f"has {EXPECTED_ROWS:,} customers",
      len(customers) == EXPECTED_ROWS, f"{len(customers):,}")

# ---- Customer IDs: the clearest single giveaway --------------------------
first_id = str(customers["customerID"].iloc[0])
ids_look_real = bool(re.fullmatch(EXPECTED_ID_PATTERN, first_id))
check("customer IDs look like 7590-VHVEG, not CUST-00000",
      ids_look_real, f"first ID is {first_id!r}")

# ---- The label -----------------------------------------------------------
churners = int((customers["Churn"] == "Yes").sum())
churn_rate = churners / len(customers)
check(f"has {EXPECTED_CHURNERS:,} churners", churners == EXPECTED_CHURNERS, f"{churners:,}")
check(f"churn rate is {EXPECTED_CHURN_RATE:.2%}",
      abs(churn_rate - EXPECTED_CHURN_RATE) < 0.005, f"{churn_rate:.2%}")

# ---- The known data-quality quirk ---------------------------------------
blank_charges = int((customers["TotalCharges"].astype(str).str.strip() == "").sum())
check(f"has exactly {EXPECTED_BLANK_CHARGES} blank TotalCharges entries",
      blank_charges == EXPECTED_BLANK_CHARGES, f"{blank_charges}")

# ---- Ranges --------------------------------------------------------------
check(f"maximum tenure is {EXPECTED_MAX_TENURE} months",
      customers["tenure"].max() == EXPECTED_MAX_TENURE,
      f"{customers['tenure'].max()}")

# ---- Billing arithmetic: the strongest tell ------------------------------
# In generated data TotalCharges is computed as tenure * MonthlyCharges, so the
# ratio is exactly 1.000 for everyone. Real billing has promotions, plan
# changes and partial months, so the ratio scatters.
total_charges = pd.to_numeric(
    customers["TotalCharges"].replace(" ", np.nan), errors="coerce")
billed = total_charges.notna() & (customers["tenure"] > 0)
ratio = total_charges[billed] / (customers["tenure"][billed] * customers["MonthlyCharges"][billed])

check("billing does not divide perfectly (real data is noisy)",
      ratio.std() > 0.01,
      f"ratio TotalCharges/(tenure*Monthly) has standard deviation "
      f"{ratio.std():.5f} - near zero means it was computed by formula")

# ---- Price list ----------------------------------------------------------
# A real telco sells a finite set of plans, so monthly charges repeat.
distinct_prices = customers["MonthlyCharges"].nunique()
check("monthly charges come from a real price list, not a random draw",
      distinct_prices < 2500,
      f"{distinct_prices:,} distinct prices among {len(customers):,} customers")

# ---- Verdict -------------------------------------------------------------
print()
print("=" * 70)
if problems:
    print(f"  {len(problems)} check(s) failed - this is NOT the IBM Telco dataset.")
    print()
    print("  Your proposal and progress report both state that the IBM Watson")
    print("  Telco Customer Churn dataset was used. Either replace this file")
    print("  with the real one, or correct that claim in the write-up.")
    print()
    print("  Real dataset:")
    print("    https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
    sys.exit(1)

print("  Verified: this is the genuine IBM Watson Telco Customer Churn dataset.")
print("=" * 70)
sys.exit(0)

# Dataset

**IBM Watson Telco Customer Churn**

`Telco-Customer-Churn.csv` — 7,043 customers, 20 features, binary churn label.

| Property | Value |
|---|---|
| Rows | 7,043 |
| Churners | 1,869 (26.54%) |
| Blank `TotalCharges` | 11 (all tenure-0 customers) |
| Tenure range | 0–72 months |
| Customer ID format | `7590-VHVEG` |

## Source

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Also mirrored by IBM at:

https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

## Verifying it

Several near-identical imitations of this dataset circulate online, and they
are not obvious on inspection — the column names and row count match, and the
marginal distributions look plausible. Before running anything:

```bash
python3 tests/verify_dataset.py
```

That script checks eight properties of the file, including the customer-ID
format, the exact churn count, and the known eleven-row `TotalCharges` gap. It
exits non-zero if any of them fail.

## A note on preprocessing

The 11 blank `TotalCharges` values belong entirely to customers with
`tenure = 0` — they had signed up but had not yet been billed. These are filled
with **0**, not with the column median. Zero is the true value: a customer
billed for zero months has been charged nothing. Median imputation would assign
these brand-new customers an average lifetime spend, which is both false and
concentrated in the group most likely to churn.

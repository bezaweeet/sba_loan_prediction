# SBA Loan Default Prediction

This project tries to answer one question: **can we tell, before a small business loan is approved, whether that loan is likely to default?**

The project uses real loan data from the U.S. Small Business Administration (SBA) to explore what makes a loan risky, and builds a few models to try to predict default before it happens.

## The Data

The dataset comes from Kaggle: [SBA 7(a) Loan Data](https://www.kaggle.com/datasets/williecosta/sba-7a-loan-data)

It contains loan records approved under the SBA's 7(a) loan program.After cleaning, this project uses **347,135 loans**, each labeled as either:
- **Paid in Full** (the loan was fully repaid), or
- **Charged Off** (the loan defaulted)

About **8% of loans in this dataset defaulted**, and 92% were paid back in full.

## What the Project Does

The work is split into a few main steps:

1. **Clean the data**: remove columns that would leak the answer (like the charge-off date), drop columns that are mostly empty, and fix data type issues.
2. **Explore the data**: look for patterns in who defaults, usingcharts broken down by state, industry, business type, loan size,and approval year.
3. **Prepare the features**: turn text categories (like state or business type) into a format a model can use, and scale numeric columns where needed.
4. **Build and compare models**: train a naive baseline, a Logistic Regression model, and a Random Forest model, then tune the Random Forest's settings using a hyperparameter search rather than guessing.
5. **Evaluate the models**: since only 8% of loans default, plain accuracy is misleading. This project uses F1 Score and PR-AUC instead, which are better suited for spotting a rare outcome.

## What was Found

A few things stood out during the exploration:

- **Location matters.** Florida, South Carolina, and Texas have the highest default rates among states with enough loans to trust the number (12.3%, 11.1%, and 11.0%).
- **Industry matters a lot.** Computer and office machine repair businesses default at 26.4%, much higher than most other industries.
- **Business type matters.** Individual borrowers default more often (9.1%) than corporations (8.0%) or partnerships (4.7%).
- **Smaller loans are riskier.** Loans that defaulted were smaller on average ($204,527) than loans that were paid back ($340,227).
- **The economy matters.** Default rates were highest in 2018 (10.65%)and lowest in 2013 (6.51%), which lines up with broader economic trends.

## Model Results

| Model | F1 Score | PR-AUC |
|---|---|---|
| Naive Baseline (always predicts "no default") | 0.0000 | 0.0804 |
| Logistic Regression | 0.3438 | 0.3716 |
| **Random Forest (tuned)** | **0.5012** | **0.5672** |

The naive baseline is included as a floor, any real model needs to beat it by a wide margin to be useful. Random Forest is the best-performing model and the one selected for this project. Its
hyperparameters (100 trees, max depth 20, and others) were chosen using a randomized search rather than guessed by hand. Compared to Logistic Regression, Random Forest catches more actual defaults (86% recall vs.82%) while flagging about half as many good loans as risky (8,728 false positives vs. 16,383).

## How to Run This

1. Open the notebook in `notebooks/` 
2. Download the dataset from the Kaggle, link above.
3. Update the file path in the notebook to point to where you saved
   the dataset.
4. Run the notebook from top to bottom.

## What's Next

Future work on this project could include testing more models (like XGBoost or LightGBM), tuning the decision threshold instead of using the default 0.5 cutoff, and checking whether the model treats different groups of borrowers fairly.

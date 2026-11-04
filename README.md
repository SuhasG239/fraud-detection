# Fraud Detection

Detecting fraudulent mobile-money transactions in the PaySim dataset
(6.36 million transactions, 0.13% fraud). The notebook covers data exploration,
feature engineering, model training, and evaluation, and saves the final model.

## Approach

- **Filter:** fraud only occurs in `TRANSFER` and `CASH_OUT` transactions, so the model
  is trained on those 2.77 million rows (0.3% fraud). Other types are treated as not fraud.
- **Features:** amount, sender and receiver balances before and after, balance
  differences, hour of day (`step % 24`), and balance-error features that measure how far
  the recorded balances are from what the amount says they should be. Account IDs and
  the rule-based `isFlaggedFraud` column are dropped.
- **Split:** 80/20, stratified so both sets keep the same fraud rate.
- **Models:** logistic regression (scaled, `class_weight='balanced'`) as a baseline,
  and histogram gradient boosting.
- **Metrics:** precision, recall, and PR-AUC on the fraud class. Accuracy is not used
  because a model that never predicts fraud is already 99.7% accurate.

## Results (test set, 554,082 transactions, 1,643 frauds)

| Model | Precision | Recall | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.044 | 0.900 | 0.978 | 0.575 |
| Gradient Boosting | 0.993 | 0.993 | 0.996 | 0.986 |

Gradient boosting catches about 99% of frauds with very few false alarms. Logistic
regression catches 90% but flags about 22 legitimate transactions for every real fraud.

PaySim is a simulated dataset, and its fraud follows a clear pattern (the sender's
account is emptied and the balances don't add up). Expect lower scores on real-world
transaction data.

## Files

- `Fraud Detection.ipynb`: exploration, feature engineering, training, and evaluation
- `fraud_model.joblib`: the trained gradient boosting model and its feature list. The
  last notebook cell shows how to load it and score a single transaction.

## Run it

1. Download the PaySim dataset from
   [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1), rename the CSV to
   `AIML Dataset.csv`, and put it in this folder. It isn't in the repo because of its
   size (about 490 MB).
2. Set up the environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   python -m ipykernel install --sys-prefix --name fraud-detection --display-name "Python (Fraud Detection)"
   ```

3. Open `Fraud Detection.ipynb` in Jupyter or VS Code, pick the
   "Python (Fraud Detection)" kernel, and run all cells. Training takes a few minutes.

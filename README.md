# Loan Default Prediction with PyTorch


## Project Overview
This project predicts the likelihood of loan defaults using machine learning.  
The goal is to help the company balance risk (avoiding bad loans) and revenue (approving good borrowers).  
The notebook (`LoanDefault.ipynb`) covers the entire workflow from EDA to model training, evaluation, and prediction.

---

## Data & Assumptions
- **Training data:** Contains historical loan applications with features (e.g., borrower attributes) and the target variable `bad_flag`.
- **Test data (`testing_loan_data.csv`):** Has the same structure as training data but with an empty `bad_flag` column. Predictions are written into this column.
- **Assumptions:**
  - Median imputation uses values from the training dataset, not the test dataset.  
  - A threshold of **0.3** is chosen to balance recall (catching bad loans) and precision (avoiding false rejections).  
  - Columns like `emp_length`, `int_rate`, and `revol_util` are cleaned before modeling.  
  - Identifiers (`id`, `member_id`) and text fields (`desc`) are dropped as they don’t add predictive value.  

---

## Techniques Used
1. **Data Preprocessing**
   - Missing value handling
   - Feature scaling and encoding
   - Train-test split
   - Conversion to PyTorch tensors

2. **Model Selection**
   - Binary classification neural network built in PyTorch.
   - Trained using binary cross-entropy loss.

3. **Model Evaluation**
   - Metrics: Recall, Precision, ROC-AUC
   - Confusion matrix to analyze trade-offs
   - Threshold tuning to balance recall and precision

---

## Model Evaluation & Threshold Choice
Different thresholds were tested to observe the trade-off between **recall (catching bad loans)** and **precision** (accuracy of predicting bad loans):  

| Threshold | Recall | Precision | ROC-AUC |
|-----------|--------|-----------|---------|
| 0.10      | 0.604  | 0.124     | 0.696   |
| 0.20      | 0.563  | 0.131     | 0.696   |
| 0.30      | 0.512  | 0.138     | 0.696   |
| 0.50      | 0.397  | 0.152     | 0.696   |

- **Low threshold (0.1–0.2):** Catches more bad loans (higher recall) but risks rejecting many good borrowers → lost revenue.
- **High threshold (0.4–0.5):** Approves more good borrowers (higher precision) but lets through more risky borrowers → higher default risk.
- **Middle ground (0.25–0.35):** Balances recall and precision.  

➡ **Chosen threshold = 0.30** as it offers a solid balance between risk control and revenue opportunities.

---

## Business Impact
- **Better risk management:** Reduces exposure to bad loans by catching defaults early.
- **Revenue protection:** Avoids rejecting too many good borrowers, maintaining customer satisfaction.
- **Operational decision-making:** Threshold tuning allows the company to adjust strategy depending on business priorities (risk-averse vs. growth-focused).  

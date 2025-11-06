# Bank Marketing Success Prediction (Cost-Sensitive Classification)

This project tackles the critical business problem of optimizing a bank's telemarketing campaign by building a machine learning model to predict client subscription to a term deposit.

The primary technical challenge is the **severe class imbalance** in the dataset (**88.3% 'No'** vs. **11.7% 'Yes'**). This project demonstrates a robust methodology for handling this imbalance through **cost-sensitive learning** and comprehensive model comparison.

## 📈 Key Results: Final Model Performance

After a rigorous comparison of 10 different algorithms, the **Tuned LightGBM (LGBM) Classifier** was selected as the best-performing model. It provides the optimal balance between finding potential customers (Recall) and not wasting resources (Precision).

| Metric | Baseline (Logistic Regression) | Tuned LightGBM (Final Model) | Business Impact |
| :--- | :--- | :--- | :--- |
| **Recall (Yes)** | 0.18 | **0.64** | **Maximize Revenue:** A 255% increase in identifying actual subscribers. |
| **Precision (Yes)** | 0.66 | **0.36** | **Efficiency:** 36% of targeted calls result in a sale (a high conversion rate). |
| **F1-Score (Yes)** | 0.28 | **0.46** | **Optimal Balance:** The best overall trade-off for this business problem. |
| **False Negatives** | 870 | ~381 | **Reduced Missed Sales:** Missed opportunities were more than halved. |

## 🛠️ Project Methodology & Technical Details

### 1. Data Integrity & Leakage Mitigation
A critical step in the preprocessing phase was identifying and mitigating data leakage.

* **Data Leakage Fix (Critical):** The **`duration` column** was **dropped** before training. A call's duration is only known *after* the client has decided to subscribe, making it a "leaky" feature that would cause artificially perfect (and useless) predictions.
* **Imbalance Strategy:** The core solution was using **Cost-Sensitive Learning**. We calculated the imbalance ratio of **7.55:1** and fed this directly into the boosting models using the `scale_pos_weight` parameter.

### 2. Model Selection (10-Model Comparison)
A comprehensive comparison of 10 algorithms proved that standard methods were ineffective against the severe imbalance. Advanced boosting frameworks were the clear winners.

| Rank | Model | Imbalance Strategy | F1-Score (Yes) |
| :--- | :--- | :--- | :--- |
| **1** | **LightGBM (LGBM)** | **Scale Weight (7.55)** | **0.46** |
| 2 | XGBoost | Scale Weight (7.55) | 0.45 |
| 3 | SVM | `class_weight='balanced'` | 0.44 |
| 4 | Gradient Boosting (GBC) | `sample_weight` | 0.43 |
| 5 | Naive Bayes (GNB) | `sample_weight` | 0.39 |

*Models 6-10 (MLP, Decision Tree, KNN, Random Forest, Logistic Regression) all scored an F1-Score of 0.33 or lower.*

### 3. Final Optimization
* **Model:** **LightGBM Classifier** was selected for its best-in-class performance and efficiency.
* **Tuning:** **`RandomizedSearchCV`** was used to find the optimal hyperparameters (e.g., `learning_rate=0.01`, `max_depth=9`, `num_leaves=50`) to maximize the **F1-Score** for the 'Yes' class.

## 📁 Project Structure

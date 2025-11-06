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

## 💻 Dependencies & How to Run

This project was built using Python 3.9+. To replicate the analysis, clone the repository and install the required dependencies.

1.  **Clone the repository:**
    ```bash
    git clone [your-github-repo-url-here]
    cd [repository-name]
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    Or, install the main libraries manually:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm jupyter
    ```

3.  **Run the notebook:**
    Launch the Jupyter environment to explore the complete workflow, from initial EDA to final model tuning.
    ```bash
    jupyter notebook Bank_Marketing_Project.ipynb
    ```

## 🚀 Future Enhancements

While the current model is highly effective, future iterations can provide even greater business value:

1.  **Model Explainability:** Implement **SHAP (SHapley Additive exPlanations)** to analyze the feature importance of the final LightGBM model. This will provide actionable business insights into *why* certain clients are more likely to subscribe (e.g., "clients contacted in May" or "clients with 'success' in previous campaigns").

2.  **Cost-Sensitive Tuning:** Develop a **custom loss function** for LightGBM that optimizes directly for **Net Profit**. This involves assigning a real financial cost to a "False Positive" (a wasted call) and a real financial gain to a "True Positive" (a successful sale), moving beyond purely statistical metrics like the F1-Score.

3.  **Advanced Resampling:** Explore the impact of advanced resampling techniques like **SMOTE-Tomek** or **ADASYN** in combination with the cost-sensitive boosting models to see if Precision can be further improved without sacrificing the high Recall.

---
**Prepared By:** Arnav kalambe

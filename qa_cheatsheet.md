# Q&A Cheat Sheet

## Why Recall over Accuracy or F1-Score?

In medical diagnosis, the costs of errors are **asymmetric**:
- **False Negative** = telling a stroke patient they're healthy → can result in **death**
- **False Positive** = flagging a healthy patient → they just get extra screening

By prioritizing Recall, I'm saying: **I'd rather have false alarms than miss a single stroke case.**

F1-Score would balance precision and recall equally, but in this domain, we explicitly value recall more because missing a stroke is catastrophic.

**The Formula:**
```
Recall = True Positives / (True Positives + False Negatives)
```

In simple terms: Of all the people who actually had strokes, what percentage did we correctly identify?

---

## Why is Naive Bayes accuracy so low (19%) if it's the best model?

This is **expected** given our goal. Naive Bayes achieves 98% Recall by being very aggressive—it flags many patients as "at risk." Most of these are false positives, which hurts accuracy.

**Think of it like airport security:** You want to catch 100% of threats, even if it means extra screening for innocent travelers.

In a clinical workflow:
1. Naive Bayes flags patients as "high risk"
2. These flagged patients then receive a secondary, more precise diagnostic test
3. False positives get cleared, true positives get treatment

**The model is a screening tool, not a final diagnosis.**

---

## What is SMOTE and why didn't you apply it to the test set?

**SMOTE = Synthetic Minority Over-sampling Technique**

It creates new synthetic examples of the minority class (stroke patients) by interpolating between existing samples.

**How it works:**
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

For each minority sample, SMOTE:
1. Finds its k-nearest neighbors
2. Randomly picks one neighbor
3. Creates a new point somewhere on the line between them

**Why only training set?**

If I applied SMOTE before splitting, synthetic versions of test samples would appear in training data. This is called **data leakage** and would give artificially inflated, invalid results.

The test set must represent **real-world distribution** (~5% stroke cases).

---

## Why didn't you use deep learning / neural networks?

Three practical reasons:

1. **Dataset size**: With only 5,000 samples, deep learning would likely **overfit**. Neural networks need tens of thousands of samples minimum.

2. **Interpretability**: In healthcare, doctors need to understand **why** a model makes a prediction. Traditional models like Decision Trees are interpretable; neural networks are "black boxes."

3. **Performance**: For tabular/structured data like this, traditional ML models (Random Forest, Gradient Boosting) often **match or exceed** neural network performance. Deep learning shines with images, text, and sequences—not tables.

---

## Which features were most predictive?

The top predictors across all my models were:

| Rank | Feature | Why it matters |
|------|---------|----------------|
| 1 | **Age** | Stroke risk increases significantly with age |
| 2 | **Average Glucose Level** | High glucose is linked to cardiovascular disease |
| 3 | **BMI** | Obesity is a known stroke risk factor |

**These align with medical literature**, which validates that my models learned meaningful, clinically relevant patterns—not just noise.

**Code to get feature importance (Random Forest):**
```python
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

for i in range(X.shape[1]):
    print(f"{X.columns[indices[i]]}: {importances[indices[i]]:.4f}")
```

---

## How would you deploy this model in a real hospital?

**Step-by-step deployment plan:**

1. Build a simple **web application** where clinicians enter patient data
2. The app sends data to the model and returns a **risk score**
3. High-risk patients get flagged for **additional clinical evaluation**
4. Use Naive Bayes as the primary screener for **maximum sensitivity**

**Key point:** The model wouldn't replace doctors—it would be a **decision support tool** to help prioritize which patients need closer attention.

**Simple Flask example:**
```python
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    patient_df = pd.DataFrame([data])
    patient_scaled = scaler.transform(patient_df)
    risk = model.predict_proba(patient_scaled)[0][1]
    return jsonify({'stroke_risk': float(risk)})
```

---

## What are the limitations of your study?

| Limitation | Impact |
|------------|--------|
| Dataset size (~5,000) | Relatively small for ML |
| Single data source | May not generalize to other populations |
| No temporal data | Can't see how features change over time |
| No external validation | Not tested on completely separate clinical data |
| Missing predictors | No cholesterol, blood pressure, or family history data |

**Being honest about limitations shows scientific rigor.**

---

## Why did you test 8 different models?

This is a **comparative analysis** project. The goal was to understand how different algorithmic approaches perform on the same problem:

| Model Type | Examples | Strength |
|------------|----------|----------|
| Linear | Logistic Regression | Simple, interpretable |
| Distance-based | KNN | Finds similar patients |
| Probabilistic | Naive Bayes | Fast, handles imbalance well |
| Tree-based | Decision Tree | Interpretable rules |
| Ensemble (Bagging) | Random Forest | Reduces variance/overfitting |
| Ensemble (Boosting) | Gradient Boosting, AdaBoost | Reduces bias |

By comparing all 8, I can make **informed recommendations** based on specific requirements (interpretability vs. performance vs. speed).

---

## How did you handle categorical variables?

Two encoding strategies:

**1. Label Encoding** — for binary variables:
```python
le = LabelEncoder()
df['ever_married'] = le.fit_transform(df['ever_married'])  # Yes/No → 1/0
```

**2. One-Hot Encoding** — for multi-category variables:
```python
df = pd.get_dummies(df, columns=['work_type', 'smoking_status'], drop_first=True)
```

**Why One-Hot?** It prevents the model from incorrectly assuming an ordinal relationship (e.g., thinking "government job" > "private job" mathematically).

---

## What would you improve with more time?

1. **Hyperparameter tuning** using `GridSearchCV` to optimize each model
2. **Ensemble voting** combining top 3 models for more stable predictions
3. **K-fold cross-validation** for more robust performance estimates
4. **Threshold optimization** — predict stroke if probability > 0.3 instead of 0.5
5. **External dataset validation** to test generalization

---

## Random Forest vs. Gradient Boosting — What's the difference?

| Aspect | Random Forest | Gradient Boosting |
|--------|---------------|-------------------|
| Training | Trees built **in parallel** (independently) | Trees built **sequentially** (each corrects previous) |
| Aggregation | **Majority vote** of all trees | **Weighted sum** of predictions |
| Bias/Variance | Reduces **variance** (overfitting) | Reduces **bias** (underfitting) |
| Overfitting | More resistant | More prone to overfit |
| Speed | Faster (parallelizable) | Slower (sequential) |

**In practice:** Gradient Boosting often gets slightly better accuracy, but Random Forest is more robust and easier to tune.

---

## What is ROC curve and AUC?

**ROC = Receiver Operating Characteristic**

It plots **True Positive Rate** (Recall) against **False Positive Rate** at various classification thresholds.

**AUC = Area Under the Curve**
- **1.0** = perfect classifier
- **0.5** = random guessing (useless)
- **> 0.8** = generally considered good

**What it shows:** The trade-off between catching more strokes (TPR) and generating more false alarms (FPR).

**Code:**
```python
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc:.2f}")
```

---

## Key Numbers to Remember

| Metric | Value |
|--------|-------|
| Highest Recall | **98%** (Naive Bayes) |
| Naive Bayes Accuracy | ~19% |
| Dataset size | ~5,100 patients |
| Class imbalance | ~5% stroke cases |
| Train/Test split | 80/20 stratified |
| Models tested | 8 |
| SMOTE result | Balanced to 50/50 in training |
| Top 3 features | Age, Glucose, BMI |

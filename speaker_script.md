# Speaker Script

## Slide 1: Title

Good afternoon everyone. My name is Amar Omerović, and today I will present my project: Comparative Analysis of Machine Learning Models for Stroke Prediction. This project explores how machine learning can help identify high-risk patients before a stroke occurs, using only basic health data.

---

## Slide 2: Problem Statement

Stroke is the second leading cause of death worldwide according to the WHO. The critical insight is that 80% of strokes are preventable with early detection. However, traditional diagnosis relies on expensive CT or MRI scans, which aren't always accessible.

My research question was: Can we predict stroke risk using simple, routinely collected patient data? This is a classic binary classification problem—we want to predict whether a patient will have a stroke or not.

---

## Slide 3: The Dataset

I used the publicly available Stroke Prediction Dataset from Kaggle, containing approximately 5,100 patient records with 11 features.

This visualization reveals our biggest challenge: severe class imbalance. Only about 5% of patients had strokes. If we built a naive model that always predicts 'No Stroke', it would achieve 95% accuracy—but it would be completely useless clinically because it would miss every stroke patient. This imbalance defined my entire engineering approach.

---

## Slide 4: Data Engineering Pipeline

To address these challenges, I built a rigorous preprocessing pipeline:

First, missing values. The BMI feature had missing values. I used mean imputation since BMI follows a roughly normal distribution.

Second, feature encoding. I applied Label Encoding for binary variables like 'Ever Married', and One-Hot Encoding for nominal categories like 'Work Type' and 'Smoking Status'.

Third, the crucial step—SMOTE. Synthetic Minority Over-sampling Technique. Instead of simply duplicating stroke cases, SMOTE generates mathematically synthesized new examples by interpolating between existing minority samples using k-nearest neighbors.

Critical constraint: SMOTE was applied only to the training set after the train-test split. Applying it before would cause data leakage and produce misleading results.

---

## Slide 5: Methodology

For my experimental design: Data Split was 80% training, 20% testing with stratified sampling to maintain class proportions.

Why I rejected Accuracy: As I mentioned, 95% accuracy is trivially achievable and meaningless here.

My chosen metric was Recall, also called Sensitivity.

In medical screening, the cost of errors is asymmetric. A False Negative—missing a stroke patient—can be fatal. A False Positive—flagging a healthy patient—just means additional testing. Therefore, I prioritized maximizing Recall—catching as many true stroke cases as possible, even at the expense of some false alarms.

---

## Slide 6: Models Implemented (Part 1)

I implemented and compared 8 different models to find the best performer.

Linear and Probabilistic Models:
- Logistic Regression — My baseline model, simple and interpretable
- K-Nearest Neighbors — Classifies based on similarity to neighboring patients; I used the elbow method to optimize K
- Support Vector Machine — Uses RBF kernel to find complex decision boundaries
- Gaussian Naive Bayes — A probabilistic model assuming feature independence—which turned out to be surprisingly effective

---

## Slide 7: Models Implemented (Part 2)

Tree-Based and Ensemble Models:
- Decision Tree — Provides clear, interpretable rules like 'If Age greater than 60 AND Glucose greater than 200, then high risk'
- Random Forest — Combines hundreds of decision trees using bagging to reduce overfitting
- Gradient Boosting — Builds trees sequentially, each correcting the previous tree's errors
- AdaBoost — Focuses on correctly classifying the hardest examples by reweighting samples

---

## Slide 8: Key Results

Here are my final results.

The clear winner for Recall was Gaussian Naive Bayes at 98%—it caught nearly every stroke case.

However, there's a significant trade-off: its accuracy was only 19%, meaning it produced many false positives.

Other models like Decision Tree at 54% and AdaBoost at 50% offered more balanced performance but missed more stroke cases.

My recommendation: For a screening tool where missing a stroke is unacceptable, Naive Bayes is the safest choice. False positives simply mean additional testing, but false negatives can mean death.

Key predictors across all models were: Age, Average Glucose Level, and BMI.

---

## Slide 9: Conclusion

To conclude:

First, we proved that stroke risk can be predicted using simple, routinely collected patient data—no expensive imaging required.

Second, by prioritizing Recall and using SMOTE to handle class imbalance, we built models that catch high-risk patients effectively.

Third, practical application: These models could serve as a first-line screening tool in hospitals, flagging patients for further evaluation.

Future work could include creating an ensemble voting classifier combining top models, validating on external clinical datasets, and developing a deployable web application for clinical use.

Thank you for your attention.

---

## Slide 10: Q&A

I am now happy to answer any questions about my methodology, implementation, or results.

---
---

# Q&A Preparation (Reference Only)

## Q1: Why did you choose Recall over Accuracy or F1-Score?
In medical diagnosis, the costs of errors are asymmetric. A false negative—telling a stroke patient they're healthy—can result in death. A false positive—flagging a healthy patient—only means they receive additional screening.

By prioritizing Recall, I'm saying: I'd rather have false alarms than miss a single stroke case. F1-Score would balance precision and recall equally, but in this domain, we explicitly value recall more.

---

## Q2: Why is accuracy so low for Naive Bayes (19%) if it's the best model?
This is expected given our goal. Naive Bayes achieves 98% Recall by being very aggressive—it flags many patients as 'at risk.' Most of these are false positives, which hurts accuracy.

Think of it like airport security: you want to catch 100% of threats, even if it means extra screening for innocent travelers. In a clinical workflow, flagged patients would receive a secondary, more precise diagnostic test.

---

## Q3: What is SMOTE and why didn't you apply it to the test set?
SMOTE stands for Synthetic Minority Over-sampling Technique. It creates new synthetic examples of the minority class by interpolating between existing samples using k-nearest neighbors.

I applied it only to the training set because applying it to the test set would cause data leakage. The test set must represent real-world data distribution—approximately 5% stroke cases.

---

## Q4: Why didn't you use deep learning / neural networks?
Three reasons:
1. Dataset size: With only 5,000 samples, deep learning would likely overfit.
2. Interpretability: In healthcare, doctors need to understand why a model makes a prediction. Traditional models are interpretable; neural networks are black boxes.
3. For tabular data like this, traditional models often match or exceed neural network performance.

---

## Q5: Which features were most predictive?
The top predictors across my models were:
1. Age — Stroke risk increases significantly with age
2. Average Glucose Level — High glucose is linked to cardiovascular disease
3. BMI — Obesity is a known stroke risk factor

These align with medical literature, validating that my models learned meaningful patterns.

---

## Q6: How would you deploy this model in a real hospital?
I would build a simple web application where clinicians enter patient data and receive a risk score. Use Naive Bayes as the primary screener for maximum sensitivity. High-risk patients would receive additional clinical evaluation. The model wouldn't replace doctors—it would be a decision support tool.

---

## Q7: What are the limitations of your study?
Key limitations:
1. Dataset size of 5,000 is relatively small
2. Single data source—may not generalize to different populations
3. No temporal data showing how features change over time
4. No external validation on completely separate clinical data
5. Missing some known stroke predictors like cholesterol or family history

---

## Q8: Why did you test 8 different models?
This is a comparative analysis project. The goal was to understand how different algorithmic approaches perform: linear vs non-linear, single models vs ensembles, probabilistic vs distance-based. By comparing all 8, I can make informed recommendations based on specific requirements.

---

## Q9: How did you handle categorical variables?
Two strategies:
1. Label Encoding for binary variables (Ever Married: Yes/No becomes 1/0)
2. One-Hot Encoding for nominal variables with multiple categories (Work Type, Smoking Status)

One-Hot Encoding prevents the model from incorrectly assuming an ordinal relationship between categories.

---

## Q10: If you had more time, what would you improve?
1. Hyperparameter tuning using GridSearchCV
2. Ensemble voting combining top models
3. K-fold cross-validation for more robust estimates
4. Threshold optimization for recall-precision trade-off
5. External dataset validation

---

## Q11: Difference between Random Forest and Gradient Boosting?
Random Forest trains trees independently in parallel on random data subsets. Final prediction by voting. Reduces variance.

Gradient Boosting trains trees sequentially, each correcting previous errors. Reduces bias but more prone to overfitting.

---

## Q12: What is ROC curve and AUC?
ROC curve plots true positive rate against false positive rate at various thresholds. AUC summarizes this as one number: 1.0 is perfect, 0.5 is random guessing. It shows the trade-off between catching more strokes and generating more false alarms.

---

## Key Numbers
- Highest Recall: 98% (Naive Bayes)
- Dataset: ~5,100 patients
- Class Imbalance: ~5% stroke cases
- Split: 80/20 stratified
- Models tested: 8
- Top Features: Age, Glucose, BMI

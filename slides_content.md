# Presentation Slides Content
*Use this text to populate your PowerPoint/Canva slides.*

---

## Slide 1: Title Slide
**Title:** Stroke Prediction: A Comparative Analysis of Machine Learning Models
**Subtitle:** Capstone Project - Intelligent Systems & Machine Learning
**Student Name:** Amar
**Date:** January 16, 2026

---

## Slide 2: Problem Statement
**The Global Challenge**
*   **Stroke Impact:** 2nd leading cause of death globally (WHO).
*   **Prevention:** 80% of strokes are preventable with early detection.
*   **The Problem:** Standard diagnosis relies on expensive imaging or late symptoms.
*   **Our Goal:** Build an ML model to predict stroke risk using basic demographic & health data.

---

## Slide 3: The Dataset
**Source & Characteristics**
*   **Source:** Healthcare Dataset (Kaggle/fedesoriano).
*   **Size:** ~5,100 Patient Records.
*   **Features (11):**
    *   *Demographics:* Gender, Age, Residence Type.
    *   *Health:* BMI, Glucose Level, Hypertension, Heart Disease.
    *   *Lifestyle:* Smoking Status, Work Type.
*   **Key Challenge:** **Class Imbalance**.
    *   Stroke Cases: ~5% (Minority).
    *   No Stroke: ~95% (Majority).

*(Recommended Image: Paste "Target Distribution" Countplot from Notebook Section 2)*

---

## Slide 4: Data Engineering Pipeline
**Preparing Data for ML**
1.  **Missing Values:** Imputed `BMI` using Mean substitution (Normal distribution assumption).
2.  **Encoding:**
    *   *Label Encoding:* Binary vars (Ever Married, Residence).
    *   *One-Hot Encoding:* Nominal vars (Work Type, Smoking).
3.  **Handling Imbalance (Crucial):**
    *   **Technique:** SMOTE (Synthetic Minority Over-sampling Technique).
    *   **Method:** Generates synthetic training examples for the minority class using K-Nearest Neighbors logic.
    *   *Constraint:* Applied to **Training Set Only** (Prevent data leakage).

---

## Slide 5: Methodology
**Experimental Design**
*   **Protocol:** 80% Training / 20% Testing (Stratified Split).
*   **Evaluation Metric:**
    *   **Accuray??** NO. (95% accuracy is trivial if we predict "No Stroke" for everyone).
    *   **Recall (Sensitivity):** YES.
    *   **Goal:** Minimize False Negatives. In medicine, missing a stroke is worse than a false alarm.

---

## Slide 6: Models Implemented (Part 1)
*Overview of 8 Models Tested*

*   **Linear & Distance:**
    1.  **Logistic Regression** (Baseline, ROC Curve analysis).
    2.  **K-Nearest Neighbors (KNN)** (Distance-based, optimized K via Elbow Method).
    3.  **Support Vector Machine (SVM)** (RBF Kernel for non-linear boundaries).
    4.  **Gaussian Naive Bayes** (Probabilistic independence).

*(Recommended Image: Paste "Logistic Regression ROC Curve" from Notebook Section 4.1)*

---

## Slide 7: Models Implemented (Part 2)
*Tree-Based & Ensembles*

*   **Trees & Boosting:**
    5.  **Decision Tree** (Interpretable rules, visualized).
    6.  **Random Forest** (Bagging ensemble, feature importance analysis).
    7.  **Gradient Boosting** (Sequential error correction).
    8.  **AdaBoost** (Adaptive weighting of misclassified samples).

*(Recommended Image: Paste "Random Forest Feature Importance" from Notebook Section 4.6)*

---

## Slide 8: Key Results
*Performance Comparison*

*(Recommended Image: Paste "Model Comparison by Recall Score" Bar Chart from Notebook Section 5.1)*

*   **Top Performer (Recall):**
    *   **Naive Bayes:** 98% Recall (Sensitivity).
    *   **Decision Tree:** 54% Recall.
    *   **AdaBoost:** 50% Recall.
*   **Trade-off Observed:**
    *   Naive Bayes achieved high Recall but low Accuracy (19%), indicating a high False Positive rate.
    *   Ensemble models provided a more balanced but less sensitive profile.
*   **Key Insight:** Prioritizing Recall ensures we catch nearly all stroke cases, though it requires secondary screening for false alarms.
*   **Feature Importance:** `Age`, `Avg Glucose Level`, and `BMI` were the strongest predictors.

---

## Slide 9: Conclusion
**Final Takeaways**
*   **Success:** Built a model capable of early warning for high-risk patients.
*   **Clinical Value:** Prioritized **Recall** to ensure patient safety.
*   **Future Work:**
    *   Combine top models into a Voting Classifier.
    *   Test on real-world external clinical data.

---

## Slide 10: Q&A
**Thank You!**
*Questions?*

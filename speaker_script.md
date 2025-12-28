# Speaker Script & Presentation Guide
*This document tells you what to **SAY** and what to **SHOW** (in the notebook) for each slide.*

---

## Slide 1: Title Slide
**SHOW:** The definition of the Introduction section in the Notebook.
**SAY:**
"Good morning everyone. My name is **Amar**, and today I will present my capstone project on **Stroke Prediction using Machine Learning**. This project focuses on identifying high-risk patients to aid in early prevention."

---

## Slide 2: Problem Statement
**SHOW:** Scroll to Section **1.1 Global Health Context**.
**SAY:**
"Stroke is the second leading cause of death worldwide. The problem is that traditional prediction often requires expensive scanning. I wanted to see if we could use simple data—like age, glucose levels, and BMI—to predict strokes before they happen. This is a classic Binary Classification problem where '1' means Stroke."

---

## Slide 3: The Dataset
**SHOW:** Scroll to **Section 2** code output where `df.shape` and the **Stroke Distribution Countplot** are shown.
**SAY:**
"I used the Stroke Prediction Dataset from Kaggle. It has about 5,000 patients and 11 features.
*Point to the graph*
As you can see here, we have a massive problem called **Class Imbalance**. Only about 5% of our patients actually had a stroke. If I just guessed 'No Stroke' for everyone, I'd be 95% correct, but I'd be useless as a doctor. This dataset challenge defined my entire engineering approach."

---

## Slide 4: Data Engineering Pipeline
**SHOW:** Scroll to **Section 2.3 Handling Class Imbalance with SMOTE** and the code block showing `SMOTE`.
**SAY:**
"To fix this, I built a rigorous pipeline:
1.  First, I handled missing BMI values by imputing the Mean.
2.  I used One-Hot Encoding for categorical jobs like 'Private' or 'Government'.
3.  **Most Importantly:** I used **SMOTE** (Synthetic Minority Over-sampling Technique).
Instead of just copying data, SMOTE mathematically generates *new* synthetic examples of stroke patients by looking at nearest neighbors. I applied this **only to the training set** to ensure my test results remain honest and valid."

---

## Slide 5: Methodology
**SHOW:** Scroll to **Section 3.2 Evaluation Metrics**.
**SAY:**
"For my methodology, I split the data 80/20.
I explicitly rejected Accuracy as my main metric. Instead, I chose **Recall (Sensitivity)**.
Why? Because in medicine, a False Negative is fatal. If I tell a stroke patient they are fine, they might die. If I tell a healthy patient they are at risk, they just get a checkup. I want to catch every single stroke, so maximizing Recall is my priority."

---

## Slide 6: Models Implemented (Part 1)
**SHOW:** Scroll to **Section 4.1 Logistic Regression** or **4.4 Naive Bayes**.
**SAY:**
"I implemented 8 models in total, divided into two categories. First, the Linear and Probabilistic approaches:
*   **Logistic Regression** served as my baseline.
*   **KNN** looked for similar patients based on distance.
*   **SVM** attempted to find a complex boundary using kernels.
*   **Naive Bayes** was tested for its probabilistic properties—which, as we'll see, turned out to be highly effective for Recall."

---

## Slide 7: Models Implemented (Part 2)
**SHOW:** Scroll to **Section 4.5 Decision Trees** or **4.6 Random Forest**.
**SAY:**
"Next, I moved to Tree-based and Ensemble models to capture non-linear patterns:
*   **Decision Tree:** This gives us clear, interpretable rules (e.g., If Age > 60).
*   **Random Forest & Gradient Boosting:** These combine hundreds of weak trees to create a powerful predictor.
*   **AdaBoost:** This focuses specifically on learning from the hard-to-classify examples."

---

## Slide 8: Key Results
**SHOW:** Scroll to the very bottom: **Section 5.1** (The Table and Bar Chart).
**SAY:**
"Here is the final showdown. As you can see in the bar chart, **Naive Bayes** produced the highest Recall at **98%**, effectively catching almost all stroke cases. However, this came at a cost—its accuracy was only **19%**, meaning it flagged many false alarms.
Other models like **Decision Tree** and **AdaBoost** scored around **50-54% Recall** with better accuracy.
For a screening tool where missing a stroke is fatal, we argue that the **High Recall of Naive Bayes** makes it the most 'medically safe' option, despite the false positives."

---

## Slide 9: Conclusion
**SHOW:** Leave the Bar Chart on screen.
**SAY:**
"In conclusion, this project proved that we can detect high-risk patients using basic data. By prioritizing Recall and using SMOTE, we successfully overcame the imbalance problem. These models could serve as an effective screening tool in hospitals to flag patients for further analysis."

---

## Slide 10: Q&A
**SAY:**
"Thank you for listening. I am happy to answer any questions about the code or parameters used."

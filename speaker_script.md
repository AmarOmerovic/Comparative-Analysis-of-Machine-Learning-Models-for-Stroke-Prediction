# Speaker Script

## Slide 1: Title

Hi everyone. I'm Amar Omerović, and I'm really excited to share my project on Stroke Prediction with you today.

Basically, I wanted to answer one big question: Can we use simple health data—stuff that doctors already collect—to spot high-risk patients *before* they actually have a stroke?

It's a straightforward goal, but as you'll see, getting there was pretty interesting. Let's dive in.


---

## Slide 2: Problem Statement

So, why focus on stroke?

Well, aside from being a major global killer, the really tragic thing about stroke is that it's vastly preventable—up to 80% of cases could be stopped if caught early.

The issue is, right now, we mostly rely on expensive scans like MRIs to diagnose it. And you can't just give everyone an MRI every year. It's too expensive and not everywhere has the machines.

So my thinking was: What if we could flag people using just the basics? Age, glucose levels, BMI. If we could build a model to do that, it could be a cheap, easy screening tool for clinics everywhere.

---

## Slide 3: The Dataset

I used a public dataset from Kaggle for this. It's got about 5,100 patients and 11 standard features like age, work type, and smoking history.

But when I first looked at the data, I spotted a huge problem immediately. Check out this chart.

See that tiny bar on the right? That's the stroke cases—only about 5% of the total.

This is what we call "severe class imbalance." And it's tricky because if I just wrote a dumb program that said "Nobody has a stroke," it would be right 95% of the time! It would look like a great model on paper, but in real life, it would be useless because it would kill people by missing every single case.

So, fixing this was my main technical challenge.

---

## Slide 4: Data Engineering Pipeline

Here's how I tackled the data issues.

First, **Missing Data**: About 200 people didn't have a BMI listed. Instead of deleting them, I filled those gaps with the average BMI. It's a safe bet since BMI usually follows a standard bell curve, and I didn't want to throw away good data.

Second, **Encoding**: Computers don't speak English, so I had to translate things like "Married" or "Smokes" into numbers.
- For simple Yes/No stuff, I just used 1s and 0s.
- For things like jobs, I used "One-Hot Encoding." That's just a fancy way of saying I gave each job its own category so the computer wouldn't accidentally think one job was "greater than" another.

Third, and most important: **SMOTE**. This was my fix for that imbalance I mentioned.
Instead of just copy-pasting the stroke cases to make the pile bigger, SMOTE actually creates *new* synthetic examples that look like the real ones. It's like drawing a dot between two existing dots.

**Crucial detail**: I only did this on my training data. If I did it on the test data, I'd be cheating.

---

## Slide 5: Methodology

For the experiment itself, I split the data 80/20. 80% to train the models, 20% to test them. And I made sure the test set kept that same 5% stroke ratio so it reflected the real world.

Now, about judging the results.

I completely ignored "Accuracy." Like I said, 95% accuracy is easy if you just predict "Healthy" for everyone.

Instead, I focused entirely on **Recall**. 
Recall just means: "Out of everyone who actually had a stroke, how many did we catch?"

In medicine, this is the only thing that matters. If I tell a healthy person they might be at risk (False Positive), they just get a checkup. No big deal. But if I tell a stroke patient they're fine (False Negative), they could die.

So my rule was: **Is it better to warn too many people than to miss one.**

---

## Slide 6: Models Implemented (Part 1)

I threw 8 different algorithms at this problem to see what would stick.

I started with the classics:
- **Logistic Regression**: The simple baseline.
- **K-Nearest Neighbors**: Groups similar patients together.
- **Support Vector Machine**: Good at drawing complex lines between groups.
- **Gaussian Naive Bayes**: A probabilistic model that's usually pretty fast.

**The ROC Curve on this slide:**
This curve is the go-to chart for medical classifiers. The vertical axis is "how many strokes we catch," the horizontal is "how many false alarms we trigger." You want the line to hug the top-left corner. The dashed diagonal is basically coin-flip guessing—our curve is above it, so the model is actually learning something.

---

## Slide 7: Models Implemented (Part 2)

Then I tried the heavy hitters—the Tree-based models:
- **Decision Tree**: Simple rules like "If Age > 60..."
- **Random Forest**: A whole bunch of trees voting together.
- **Gradient Boosting & AdaBoost**: These learn from their mistakes to get better over time.

**The Feature Importance chart on this slide:**
Doctors want to know what's driving the prediction. This bar chart shows exactly that. Each bar = one feature. Taller bar = more important. Age is the biggest one—makes sense, older people have more strokes. Glucose and BMI are next. This matches real medical research, so we know the model is picking up on real patterns.

---

## Slide 8: Key Results

But here's the surprise.

The winner? **Gaussian Naive Bayes.**

It hit a **98% Recall**. That means out of 100 stroke patients, it found 98 of them. It missed almost no one.

**The Comparison Chart on this slide:**
This chart puts all 8 models next to each other. Naive Bayes is way taller than everything else—it caught 98% of strokes. Decision Tree is second at 54%. Most others are even lower.

Now the catch: Naive Bayes only has 19% accuracy. Sounds terrible, but think about it—that just means it flags a lot of healthy people too. In a real hospital, those people would just get a second check. No harm done. But we almost never miss an actual stroke patient. That's what matters.

So, Naive Bayes acts like a sensitive metal detector—it beeps a lot, but it never misses a weapon.

---

## Slide 9: Conclusion

So, what's the takeaway?

1. **Simple data works.** We don't necessarily need MRIs to start screening people.
2. **Context is everything.** If I had just optimized for accuracy, I would have built a useless model. By chasing Recall, I built something that could actually save lives.
3. **The "Best" model isn't always the smartest one.** The simplest probabilistic model beat the complex neural-style ones because it fit our specific safety-first strategy better.

If I were to take this further, I'd probably try to combine these models to get the best of both worlds—keep that high recall but maybe reduce the false alarms a bit.

Thanks for listening. I'd love to answer any questions.

---

## Slide 10: Q&A

(I'm ready for your questions!)

---
---

# Code Implementation Walkthrough

I built this project as a Jupyter Notebook so I could run code step-by-step and visualize results immediately. Let me walk you through each section.

**Step 1: Import libraries**

I started by importing the standard tools. Scikit-learn is the industry standard for machine learning in Python because it provides a consistent interface for every algorithm.
```python
import pandas as pd              # For loading the CSV and managing dataframes
import numpy as np               # For numerical operations
import matplotlib.pyplot as plt  # For creating static plots
import seaborn as sns            # For creating nicer statistical visualizations

# Machine Learning Core
from sklearn.model_selection import train_test_split  # Critical for valid testing
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Transformations
from sklearn.impute import SimpleImputer  # For filling missing BMI values
from sklearn.metrics import recall_score, accuracy_score  # Our key metrics

# The Solution for Imbalance
from imblearn.over_sampling import SMOTE  # Generates synthetic samples
```

**Step 2: Load and explore the data**

I loaded the dataset and immediately removed the 'ID' column. ID is just a database index. If I leave it in, the model might cheat by memorizing that 'Patient 5002 had a stroke' instead of learning real patterns like age or glucose.
```python
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Remove ID to prevent "memorization" (overfitting)
df = df.drop('id', axis=1)
```

**Step 3: Handle missing values**

About 200 patients had missing BMI data. Instead of deleting them (which loses data), I used Mean Imputation. Since BMI follows a normal bell curve in the population, the average is a safe, unbiased estimate to fill in the gaps.
```python
imputer = SimpleImputer(strategy='mean')
df['bmi'] = imputer.fit_transform(df[['bmi']])
```

**Step 4: Encode categorical variables**

Machines can't read text like 'Married' or 'Smokes'; they only understand numbers. But I had to be careful how I converted them:

**For Binary Variables (Yes/No):**
I used Label Encoding, which just turns Yes/No into 1/0. Simple and effective.
```python
le = LabelEncoder()
df['ever_married'] = le.fit_transform(df['ever_married'])
```

**For Multi-Category Variables (Jobs, Smoking):**
Here I used One-Hot Encoding. If I just numbered jobs 1, 2, 3, the model would think Job 3 is 'greater than' Job 1, which is false mathematically. One-Hot creating a separate True/False column for each job prevents this error.
```python
df = pd.get_dummies(df, columns=['gender', 'work_type', 'smoking_status'], drop_first=True)
```

**Step 5: Split data BEFORE applying SMOTE (CRITICAL)**

This is the most critical step for validity. I used `stratify=y` to force the Test Set to have the exact same 5% stroke rate as reality. Without this, a random split might result in a Test Set with zero strokes, making evaluation impossible.
```python
X = df.drop('stroke', axis=1)  # Features
y = df['stroke']               # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

**Step 6: Apply SMOTE to training data ONLY**

This is where I handle the imbalance. I apply SMOTE to generate synthetic stroke cases, but ONLY on the training data. If I applied it to the whole dataset first, synthetic copies of test patients would leak into the training set, and my high scores would be fake (Data Leakage).
```python
smote = SMOTE(random_state=42)

# Fit ONLY on X_train. The Test set remains pure and untouched.
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

**Step 7: Scale features**

Algorithms like K-Nearest Neighbors and Support Vector Machine calculate 'distance' between patients. If Age is 0-80 and Glucose is 0-300, Glucose would dominate the distance calculation just because the numbers are bigger. Standardization forces all features onto the same scale so they contribute equally.
```python
scaler = StandardScaler()

# Learn the scale from TRAINING data only...
X_train_scaled = scaler.fit_transform(X_train_resampled)

# ...and apply that same scale to Testing data. 
X_test_scaled = scaler.transform(X_test)
```

**Step 8: Train and evaluate**

Finally, the training loop. I used scikit-learn's consistent 3-step pattern: Initialize, Fit, Predict. I focus on Recall score because, in this domain, a False Negative is deadly.
```python
# 1. Initialize
nb_model = GaussianNB()

# 2. Fit (Train on the Balanced Training Set)
nb_model.fit(X_train_scaled, y_train_resampled)

# 3. Predict (Test on the Imbalanced, Real-World Test Set)
y_pred_nb = nb_model.predict(X_test_scaled)

# 4. Evaluate
print(f"Recall: {recall_score(y_test, y_pred_nb)}")
```

---
---

# Notebook Chart Explanations (If Asked to Open Notebook)

Use these explanations if you need to open the Jupyter Notebook and explain additional charts not shown on the slides.

---

## Target Distribution Countplot (Section 2)
This shows the imbalance problem. One bar is huge (healthy patients, ~95%), the other is tiny (stroke patients, ~5%). If we just guessed "no stroke" every time, we'd be right 95% of the time—but totally useless.

**What to point out:** "See how lopsided this is? This is why accuracy is a trap. A lazy model would just predict 'no stroke' for everyone and look great on paper. That's why I focused on Recall instead."

---

## Correlation Heatmap (Section 2)
This grid shows which features are related to each other. Darker squares = stronger connection. I used it to see which features connect most to having a stroke. Also helps spot redundant features.

**What to point out:** "Look at the 'stroke' row at the bottom—you can see which features have the darkest squares next to it. Age is one of them. This confirmed that age is our strongest predictor."

---

## Age Distribution Histogram (Section 2)
Shows the spread of ages in the data. We have patients from young to old, which is good because age is the #1 stroke predictor.

**What to point out:** "The dataset has a good mix of ages. That's important because if we only had 30-year-olds, the model couldn't learn that older people are at higher risk."

---

## BMI Distribution (Section 2)
BMI forms a nice bell curve centered around 28-29. That's why filling missing BMI values with the average was a safe choice—it doesn't mess up the shape.

**What to point out:** "See this bell shape? That's a normal distribution. When data looks like this, using the average to fill gaps is a standard, safe choice. It doesn't skew anything."

---

## Glucose Level Distribution (Section 2)
Most people have normal glucose (under 140), but there's a long tail of high-glucose patients. Those outliers are important since high glucose links to heart problems and stroke.

**What to point out:** "Most patients cluster on the left (normal glucose). But see that tail stretching to the right? Those are the high-glucose patients, and they're at higher stroke risk."

---

## Confusion Matrix (Per Model)
A 2x2 box:
- **Top-Left**: Healthy people we correctly said are healthy
- **Top-Right**: Healthy people we wrongly flagged (false alarms—annoying but not deadly)
- **Bottom-Left**: Stroke patients we MISSED (bad—this is what we want to minimize)
- **Bottom-Right**: Stroke patients we correctly caught

For Naive Bayes, the bottom-left box is nearly empty—that's why Recall is so high.

**What to point out:** "The bottom-left cell is the danger zone—those are stroke patients we missed. With Naive Bayes, that number is almost zero. That's why I picked it as the winner."

---

## ROC Curves (Per Model)
Plots "strokes caught" vs "false alarms" at every possible threshold. The closer to the top-left, the better. AUC (Area Under Curve) summarizes it: 1.0 = perfect, 0.5 = random guessing.

**What to point out:** "The dashed diagonal line is what random guessing would look like. Our curve is above it, which means the model is actually learning, not just flipping a coin."

---

## Precision-Recall Curve (If Present)
Shows the trade-off between "how accurate are our alarms" vs "how many strokes do we catch." With Naive Bayes, catching more strokes means way more false alarms—but we accept that.

**What to point out:** "This curve drops steeply—meaning if we want to catch more strokes, we're going to trigger a lot more false alarms. In medicine, that trade-off is worth it."

---

## Learning Curves (If Present)
Shows how the model improves as it sees more training data. If the training line is way higher than validation, the model is memorizing instead of learning (overfitting).

**What to point out:** "If there's a big gap between the two lines, the model is overfitting—basically memorizing the training data instead of learning patterns. Ideally, both lines should meet."

---

## Decision Tree Visualization (Section 4.5)
A flowchart of the tree's logic. Start at the top, follow the branches ("Age > 60? Yes/No") until you hit a leaf that says "Stroke" or "No Stroke." Fully transparent—you can trace exactly why any patient got flagged.

**What to point out:** "This is why decision trees are so interpretable. I can literally trace any patient from top to bottom and explain exactly why the model made its decision. Try doing that with a neural network!"

---

## Key Numbers
- Highest Recall: 98% (Naive Bayes)
- Dataset: ~5,100 patients
- Class Imbalance: ~5% stroke cases
- Split: 80/20 stratified
- Models tested: 8
- Top Features: Age, Glucose, BMI

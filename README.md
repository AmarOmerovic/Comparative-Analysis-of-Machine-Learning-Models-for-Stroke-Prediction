# Stroke Prediction: A Comparative Analysis of Machine Learning Models

## Project Overview
This capstone project aims to predict the likelihood of a patient having a stroke based on various demographic, lifestyle, and clinical factors. It involves a rigorous comparative analysis of eight different machine learning algorithms, prioritizing **Recall (Sensitivity)** to minimize false negatives in a medical context.

## Dataset
The project uses the **Stroke Prediction Dataset** (originally from Kaggle/fedesoriano), containing over 5,000 patient records.
- **File:** `healthcare-dataset-stroke-data.csv`
- **Target Variable:** `stroke` (1 = Stroke, 0 = No Stroke)
- **Challenge:** Severe class imbalance (~5% positive cases).

## Models Evaluateds
1.  Logistic Regression
2.  K-Nearest Neighbors (KNN)
3.  Support Vector Machine (SVM)
4.  Gaussian Naive Bayes
5.  Decision Tree
6.  Random Forest
7.  Gradient Boosting
8.  AdaBoost

## Key Findings
- **Class Imbalance Handling:** SMOTE (Synthetic Minority Over-sampling Technique) was applied to the training set to address the imbalance.
- **Best Performer:** **Gaussian Naive Bayes** achieved the highest Recall (~98%), making it the most suitable model for screening purposes where missing a positive case is critical.
- **Ensemble Methods:** Random Forest and Gradient Boosting showed strong balanced performance (high Accuracy and AUC).

## Project Structure
- `stroke_prediction_capstone.ipynb`: The main Jupyter Notebook containing the full analysis, code, and visualizations.
- `generate_notebook.py`: Python script used to programmatically generate the notebook structure.
- `slides_content.md` & `speaker_script.md`: Presentation materials for the project defense.
- `healthcare-dataset-stroke-data.csv`: Source dataset.

## Setup and Usage

### Prerequisites
- Python 3.12+
- Jupyter Notebook

### Installation
1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    (Ensure you have `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn`, `jupyter`, `notebook` installed)
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter notebook
    ```

### Running the Notebook
1.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
2.  Open `stroke_prediction_capstone.ipynb`.
3.  Run all cells to reproduce the analysis and results.

## Author
Amar Omerovic

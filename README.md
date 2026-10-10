# Bank Marketing Machine Learning Project

## Overview

This project uses the **UCI Bank Marketing dataset** to investigate whether customer demographic, financial, and campaign-related information can be used to predict whether a customer will subscribe to a term deposit.

This repository is an initial skeleton for a 3-person CS 5821 Machine Learning course project.

## Domain Problem

Banks conduct direct-marketing campaigns to identify customers who may be interested in financial products.

### Initial research question

**Can we predict whether a customer will subscribe to a term deposit using customer, financial, and campaign-related features?**

A secondary goal is to identify which variables are most useful for prediction and compare several classical machine-learning classifiers.

## Primary Dataset

**UCI Machine Learning Repository — Bank Marketing**

Source:  
https://archive.ics.uci.edu/dataset/222/bank+marketing

The commonly used `bank-full.csv` version contains approximately **45,000 observations** and a mixture of:

- Numerical features
- Binary categorical features
- Multi-class categorical features
- Binary target variable (`y`: yes/no)

See [`data/README.md`](data/README.md) for dataset notes.

## Potential Machine Learning Methods

Possible models include:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis (QDA)
- Naive Bayes

The exact set of models may change after exploratory analysis.

## Potential Evaluation Metrics

Because this is a binary classification problem, possible metrics include:

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix
- ROC curve
- ROC-AUC
- Cross-validation performance

## Initial Workstreams

The team can work in parallel on three initial areas.

### 1. Dataset Understanding and EDA
- Review the data dictionary
- Inspect feature distributions
- Analyze class balance
- Identify missing, unknown, or unusual values
- Explore predictor relationships
- Investigate possible data leakage

### 2. Preprocessing
- Define train/test split
- Encode categorical variables
- Scale numerical variables where appropriate
- Handle missing/unknown values
- Build a reusable `scikit-learn` preprocessing pipeline

### 3. Modeling and Evaluation
- Build a baseline classifier
- Set up a reusable training/evaluation workflow
- Add candidate classifiers
- Compare models using consistent metrics
- Produce confusion matrices and ROC-AUC results

## Repository Structure

```text
bank-marketing-ml-project/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
└── results/
    └── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone <repo-url>
cd bank-marketing-ml-project
```

### 2. Create a virtual environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the Bank Marketing dataset from UCI and place the local CSV file inside `data/`.

The repository ignores CSV/ZIP dataset files by default so that the source data is not unnecessarily committed to Git.

## Team Notes

The current repository is intentionally minimal.

After the initial team discussion, the group should finalize:

1. Exact research question
2. Feature inclusion/exclusion rules
3. Data-leakage policy
4. Train/validation/test strategy
5. Model comparison plan
6. Team responsibilities
7. Branching / pull-request workflow

## Course Concepts Potentially Used

Depending on the final scope, the project may use:

- Statistical learning
- Prediction vs. inference
- Training vs. test error
- Bias-variance trade-off
- KNN
- Logistic regression
- Qualitative predictors / dummy variables
- LDA / QDA
- Naive Bayes
- Feature transformations
- `scikit-learn` pipelines
- Cross-validation

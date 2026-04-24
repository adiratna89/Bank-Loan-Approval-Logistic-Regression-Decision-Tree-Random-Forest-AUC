# 🏦 Bank Loan Approval Prediction

## A Comparative Analysis of Logistic Regression, Decision Tree & Random Forest with AUC-ROC

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-green?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Made with ❤️](https://img.shields.io/badge/Made%20with-❤️-red)](https://github.com/adiratna89)

---

## 📌 Project Overview

This project builds and compares three machine learning classification models to predict whether a bank loan application should be **approved** or **rejected**. The goal is to help financial institutions automate and improve their loan approval decisions, reducing risk and speeding up processing time.

---

## 🎯 Objective

- Perform **Exploratory Data Analysis (EDA)** on the bank loan dataset.
- Build three classification models:
  - ✅ **Logistic Regression**
  - ✅ **Decision Tree Classifier**
  - ✅ **Random Forest Classifier**
- Evaluate and compare all models using **Accuracy** and **AUC-ROC** scores.
- Identify the **best-performing model** for loan approval prediction.

---

## 📊 Dataset

| Attribute | Description |
|---|---|
| **Source** | [Kaggle - Bank Loan Approval Dataset](https://www.kaggle.com/datasets/vikramamin/bank-loan-approval-lr-dt-rf-and-auc) |
| **Records** | 5,000 customer records |
| **Features** | 14 input features |
| **Target** | `Personal.Loan` — Binary (0 = Not Approved, 1 = Approved) |

### Features Included:

| Feature | Description |
|---|---|
| `ID` | Customer ID (dropped during preprocessing) |
| `Age` | Age of the customer (years) |
| `Experience` | Professional experience (years) |
| `Income` | Annual income (in thousands) |
| `ZIP.Code` | Postal code (dropped during preprocessing) |
| `Family` | Family size (1–4) |
| `CCAvg` | Average monthly credit card spending |
| `Education` | Education level (1=Undergrad, 2=Graduate, 3=Advanced) |
| `Mortgage` | Value of house mortgage |
| `Personal.Loan` | **Target** — Did the customer take a personal loan? (0/1) |
| `Securities.Account` | Does the customer have a securities account? (0/1) |
| `CD.Account` | Does the customer have a CD account? (0/1) |
| `Online` | Does the customer use internet banking? (0/1) |
| `CreditCard` | Does the customer have a bank credit card? (0/1) |

---

## 🔍 Exploratory Data Analysis (EDA) Highlights

- **Target Distribution:** Highly imbalanced — ~90.4% Not Approved (0), ~9.6% Approved (1) |
- **Key Observations:**
  - `CD.Account`, `Securities.Account`, and `Personal.Loan` are heavily skewed toward 0.
  - `Online` and `CreditCard` show more balanced distributions.
  - Higher education and income levels correlate positively with loan approval.
  - Customers with CD Accounts and Securities Accounts are more likely to be approved.

---

## 🧠 Models Built

### 1. Logistic Regression
- Baseline linear model for binary classification.
- Provides interpretable coefficients showing feature impact.
- Uses L2 regularization (default).

### 2. Decision Tree Classifier
- Non-linear model that learns decision rules from data.
- Captures feature interactions naturally.
- Visualized the full tree structure for interpretability.

### 3. Random Forest Classifier
- Ensemble of multiple decision trees.
- Reduces overfitting compared to a single Decision Tree.
- Uses bagging and feature randomness for robustness.

---

## 📈 Model Evaluation & Results

| Model | Accuracy | True Positives | False Positives | False Negatives | True Negatives |
|---|---|---|---|---|---|
| **Logistic Regression** | 95.4% | 71 | 12 | 34 | 883 |
| **Decision Tree** | 98.8% | 98 | 5 | 7 | 890 |
| **Random Forest** ⭐ | **99.1%** | **97** | **1** | **8** | **894** |

> ⭐ **Random Forest** achieved the **highest accuracy (99.1%)** and the best balance of precision and recall, making it the best model for this task.

---

## 📉 ROC-AUC Comparison

The **Area Under the ROC Curve (AUC)** was used to evaluate ranking performance:

| Model | AUC Score |
|---|---|
| Logistic Regression | ~0.92 |
| Decision Tree | ~0.96 |
| **Random Forest** | **~0.99** |

AUC values confirm that **Random Forest** consistently outperforms the other models in distinguishing between approved and non-approved loans.

---

## 🗂️ Project Structure

```
Bank-Loan-Approval-Logistic-Regression-Decision-Tree-Random-Forest-AUC/
│
├── Dataset/                     # Raw dataset (bankloan.csv)
├── Output Images/               # Saved plots (ROC curves, confusion matrices, etc.)
├── Bank Loan Approval - Logistic Regression, Decision Tree, Random Forest and AUC.ipynb
│                                # Complete analysis notebook
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook

### Installation

```bash
# Clone the repository
git clone https://github.com/adiratna89/Bank-Loan-Approval-Logistic-Regression-Decision-Tree-Random-Forest-AUC.git

# Navigate into the project folder
cd Bank-Loan-Approval-Logistic-Regression-Decision-Tree-Random-Forest-AUC

# Install required packages
pip install -r requirements.txt
```

### Usage

```bash
# Open Jupyter Notebook
jupyter notebook

# Open and run all cells in:
# Bank Loan Approval - Logistic Regression, Decision Tree, Random Forest and AUC.ipynb
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

---

## 🔑 Key Findings

1. **Random Forest** is the best model for this loan approval prediction task, achieving **99.1% accuracy**.
2. Features like `Income`, `Education`, `CCAvg`, and `Family` have the strongest influence on loan approval.
3. The dataset is **highly imbalanced** — only ~9.6% of applicants were approved, which is realistic for real-world loan scenarios.
4. **AUC-ROC** is a more reliable metric than accuracy alone for imbalanced classification problems.

---

## 📌 Future Improvements

- [ ] Implement **SMOTE** or **undersampling** to handle class imbalance more effectively.
- [ ] Perform **hyperparameter tuning** using `GridSearchCV` for all three models.
- [ ] Add **feature importance visualization** for Random Forest.
- [ ] Explore **XGBoost** and **Gradient Boosting** for comparison.
- [ ] Deploy the best model as a **web app** using Flask or Streamlit.
- [ ] Add **cross-validation** for more robust evaluation.

---

## 🤝 Contributing

Feel free to fork this repository, suggest improvements, or submit a pull request!

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Adiratna Kamble**

[![GitHub](https://img.shields.io/badge/GitHub-@adiratna89-black?logo=github&logoColor=white)](https://github.com/adiratna89)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Adiratna%20Kamble-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/adiratna89)

---

> 🎓 *This project was created as part of my Machine Learning learning journey. Feedback and suggestions are always welcome!*

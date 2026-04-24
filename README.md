# 🏦 Bank Loan Approval Prediction

## A Comparative Analysis of Logistic Regression, Decision Tree & Random Forest with AUC-ROC

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Pandas 2.0+](https://img.shields.io/badge/Pandas-2.0+-green?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn 1.2+](https://img.shields.io/badge/scikit--learn-1.2+-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Model Accuracy 99.1%](https://img.shields.io/badge/Best%20Accuracy-99.1%25-brightgreen)](https://github.com/adiratna89/Bank-Loan-Approval-Logistic-Regression-Decision-Tree-Random-Forest-AUC)
[![License MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Made with ❤️](https://img.shields.io/badge/Made%20with-%E2%9D%A4%EF%B8%8F-red)](https://github.com/adiratna89)

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [Models Built](#-models-built)
- [Model Evaluation & Results](#-model-evaluation--results)
- [ROC-AUC Comparison](#-roc-auc-comparison)
- [Key Findings](#-key-findings)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Requirements](#-requirements)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## 📌 Project Overview

This project builds and compares three machine learning classification models to predict whether a bank loan application should be **approved** or **rejected**. The goal is to help financial institutions automate and improve their loan approval decisions, reducing risk and speeding up processing time.

### 🎯 Objective

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
|-----------|-------------|
| **Source** | [Kaggle - Bank Loan Approval Dataset](https://www.kaggle.com/datasets/vikramamin/bank-loan-approval-lr-dt-rf-and-auc) |
| **Records** | 5,000 customer records |
| **Features** | 14 input features |
| **Target** | `Personal.Loan` — Binary (0 = Not Approved, 1 = Approved) |

### Features Included:

| Feature | Description |
|---------|-------------|
| ID | Customer ID (dropped during preprocessing) |
| Age | Age of the customer (years) |
| Experience | Professional experience (years) |
| Income | Annual income (in thousands) |
| ZIP.Code | Postal code (dropped during preprocessing) |
| Family | Family size (1–4) |
| CCAvg | Average monthly credit card spending |
| Education | Education level (1=Undergrad, 2=Graduate, 3=Advanced) |
| Mortgage | Home mortgage value (in thousands) |
| Personal.Loan | Accepted personal loan in previous campaign (target) |
| Securities.Account | Holds securities account (0/1) |
| CD.Account | Holds certificate of deposit (0/1) |
| Online | Uses internet banking (0/1) |
| CreditCard | Holds bank credit card (0/1) |

---

## 🔍 Exploratory Data Analysis (EDA)

### Target Variable Distribution

![Class Distribution](Output%20Images/Countplot_Personal.Loan.png)

The dataset is **imbalanced** — approximately 90.4% of customers did not accept a personal loan, while only 9.6% did. This is typical in real-world banking scenarios.

### Age Distribution

![Age Distribution](Output%20Images/Age%20chart%20of%20dataset.png)

Age distribution shows most customers fall between 25–65 years, with a concentration around 45–55 years.

### Key Feature Insights

| Feature | Insight |
|---------|--------|
| **Education** | Higher education (Graduate/Advanced) correlates with higher loan approval |
| **Income** | Higher income customers are more likely to accept loans |
| **CCAvg** | Customers with higher credit card spending show more interest |
| **CD.Account** | Holding a CD account is a strong positive indicator |
| **Securities.Account** | Having a securities account increases approval likelihood |
| **Family** | Larger family sizes show slightly higher approval rates |

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

### Accuracy Comparison

| Model | Accuracy | True Positives | False Positives | False Negatives | True Negatives |
|-------|----------|----------------|-----------------|-----------------|----------------|
| **Logistic Regression** | 95.4% | 71 | 12 | 34 | 883 |
| **Decision Tree** | 98.8% | 98 | 5 | 7 | 890 |
| **Random Forest** ⭐ | **99.1%** | **97** | **1** | **8** | **894** |

### Model Result Visualizations

| Logistic Regression | Decision Tree |
|---------------------|---------------|
| ![Logistic Regression Results](Output%20Images/Result%20of%20Logistic%20Regression%20Algorithm.png) | ![Decision Tree Results](Output%20Images/Result%20of%20Decision%20Tree%20Algorithm.png) |

| Random Forest (Best Model) |
|----------------------------|
| ![Random Forest Results](Output%20Images/Result%20of%20Random%20Forest%20Algorithm.png) |

---

## 📉 ROC-AUC Comparison

The **Area Under the ROC Curve (AUC)** was used to evaluate ranking performance:

| Model | AUC Score |
|-------|-----------|
| Logistic Regression | **0.968** |
| Decision Tree | **0.964** |
| **Random Forest** | **0.999** |

![ROC Curve Comparison](Output%20Images/AUC_ROC_CURVE.png)

**Random Forest** achieved the highest AUC score of **0.999**, indicating near-perfect discrimination between approved and non-approved loan applications.

---

## 🔑 Key Findings

1. **Random Forest is the best model** with **99.1% accuracy** and **0.999 AUC**.
2. **Most important features**: Income, Education, CCAvg, and Family size.
3. **Class imbalance** (90.4% vs 9.6%) reflects real-world banking data.
4. **Decision Tree** slightly outperforms Logistic Regression in AUC despite higher accuracy.
5. **Ensemble methods** (Random Forest) significantly reduce false positives.

---

## 🗂️ Project Structure

```
Bank-Loan-Approval-Logistic-Regression-Decision-Tree-Random-Forest-AUC/
├── Dataset/
│   └── bankloan.csv              # Raw dataset
├── Output Images/
│   ├── AUC_ROC_CURVE.png         # ROC curves for all 3 models
│   ├── Age chart of dataset.png  # Age distribution
│   ├── Countplot_*.png           # EDA feature distributions
│   └── Result_*.png              # Confusion matrices
├── .gitignore
├── README.md
├── requirements.txt
└── Bank Loan Approval - Logistic Regression, Decision Tree, Random Forest and AUC.ipynb
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/adiratna89/Bank-Loan-Approval-Logistic-Regression-Decision-Tree-Random-Forest-AUC.git
cd Bank-Loan-Approval-Logistic-Regression-Decision-Tree-Random-Forest-AUC
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Notebook

```bash
jupyter notebook
```

Open `Bank Loan Approval - Logistic Regression, Decision Tree, Random Forest and AUC.ipynb` and run all cells.

---

## 📦 Requirements

| Package | Version |
|---------|--------|
| pandas | >= 2.0.0 |
| numpy | >= 1.24.0 |
| scikit-learn | >= 1.2.0 |
| matplotlib | >= 3.7.0 |
| seaborn | >= 0.12.0 |
| jupyter | >= 1.0.0 |

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📌 Future Improvements

- [ ] Implement **GridSearchCV** for hyperparameter tuning.
- [ ] Add more algorithms: **XGBoost, LightGBM, SVM, KNN**.
- [ ] Perform **feature engineering** and selection.
- [ ] Apply **SMOTE** for handling class imbalance.
- [ ] Build a **Flask/Streamlit web app** for real-time predictions.
- [ ] Deploy the model using **Heroku or AWS**.
- [ ] Add **cross-validation** for robust evaluation.

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Adiratna Kamble**

[![GitHub](https://img.shields.io/badge/GitHub-@adiratna89-black?logo=github&logoColor=white)](https://github.com/adiratna89)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Adiratna%20Kamble-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/adiratna89)

---

<div align="center">

**If you found this project helpful, please give it a ⭐ Star!**

Made with ❤️ by Adiratna Kamble

</div>

# Titanic Survival Prediction — A Cross-Validated Machine Learning Pipeline

## 📌 Project Overview
This project addresses the classic **Titanic survival prediction** problem using a clean, well-validated machine learning pipeline. Rather than focusing on leaderboard tricks, the emphasis is on **good ML practices** such as feature engineering, cross-validation discipline, and data leakage prevention.

The project is designed as a **portfolio-quality example** demonstrating how to approach a real-world tabular classification problem responsibly.

---

## 🎯 Problem Statement
Given passenger demographic and travel information, the goal is to predict whether a passenger survived the Titanic disaster.

Key objectives of this project:
- Build a reproducible ML pipeline
- Apply meaningful feature engineering
- Use cross-validation as the primary decision metric
- Avoid data leakage

---

## 📊 Dataset
- Source: Kaggle Titanic Competition
- Target variable: `Survived` (binary classification)
- Data types:
  - **Numerical**:  SibSp, Parch, Pclass
  - **Categorical**: Sex, Embarked, Cabin, Name, Ticket

---

## 🛠 Feature Engineering
Feature engineering is a core focus of this project. The following features were created with domain reasoning:

- **Title** (extracted from `Name`)
  - Captures social status and approximate age group
- **FamilySize** (`SibSp + Parch + 1`)
  - Represents whether a passenger was traveling alone or with family
- **IsAlone** (binary)
  - Simplified signal derived from FamilySize
- **HasCabin** (extracted from `Cabin`)
  - Simplified signal derived from Cabin

Rare category (titles) is grouped to reduce noise and overfitting.

---

## 🧹 Missing Value Handling
- **Embarked**: mode imputation

All imputations are performed **inside the add_features method** to prevent data leakage.

---

## 🔁 Preprocessing Pipeline
A pipeline is used to ensure:
- Separation of numeric and categorical features
- Leakage-free cross-validation

This approach guarantees reproducibility and safe model evaluation.

---

## 📈 Model & Validation Strategy
- **Model**: CatBoostClassifier
- **Validation**: Stratified K-Fold Cross-Validation
- **Metric**: Accuracy

Model selection and feature decisions were guided strictly by cross-validation results. The Kaggle leaderboard was used only as secondary confirmation.

---

## ❌ What Didn’t Work
Several approaches were tested and intentionally discarded:
- Naive ensembling and hard voting
- Over-aggressive feature engineering
- Leaderboard-driven hyperparameter tuning

These methods either failed to improve cross-validation performance or introduced instability.

---

## 🏁 Final Results
- **Cross-validation accuracy**: ~0.83 ± 0.01
- **Kaggle public leaderboard score**: ~0.80

The close alignment between CV and leaderboard scores indicates good generalization.

---

## 🧠 Key Learnings
- Feature quality matters more than model complexity
- Cross-validation discipline is critical for generalization
- Pipelines are essential for preventing data leakage
- Simple, well-validated models often outperform naive ensembles

---

## 🔮 Future Work
- Proper out-of-fold (OOF) stacking
- Model interpretability using SHAP
- Testing the pipeline on other tabular datasets

---

## 📎 Notes
This project is intended as a **learning-focused portfolio example**, demonstrating sound machine learning practices rather than leaderboard optimization.
Also clone the repository and run notebooks locally.
Data is accessed via relative paths.


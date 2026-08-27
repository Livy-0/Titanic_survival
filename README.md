# Titanic Survival Prediction — Data Science Capstone

A beginner data science project that explores Titanic passenger data and uses machine learning to predict whether a passenger survived.

---

## 1. Problem Statement

The main question I focused on was:

> **What features contributed to the death and survival of a passenger?**

The goal of this project is to predict whether a Titanic passenger survived based on information such as passenger class, sex, age, family relationships, fare, and port of embarkation.

This project applies a complete data science workflow, from data cleaning and exploration to visualization, machine learning, and model evaluation.

---

## 2. Dataset

The dataset contains 891 Titanic passenger records.

* **Dataset Source:** [Titanic Dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)

### Key Columns

| Column | Description |
| :--- | :--- |
| **PassengerId** | Unique passenger identification number |
| **Survived** | Survival status: `0` = did not survive, `1` = survived |
| **Pclass** | Passenger class: `1st`, `2nd`, or `3rd` |
| **Name** | Passenger name |
| **Sex** | Passenger sex |
| **Age** | Passenger age |
| **SibSp** | Number of siblings/spouses aboard |
| **Parch** | Number of parents/children aboard |
| **Ticket** | Ticket number |
| **Fare** | Passenger fare |
| **Cabin** | Cabin number |
| **Embarked** | Port of embarkation |

---

## 3. My Approach

### Data Cleaning
* Checked the dataset for missing values and handled them prior to analysis.
* Filled missing `Age` values using the **median age**.
* Filled missing `Embarked` values using the **most frequent value (mode)**.
* Dropped the `Cabin` column due to a large proportion of missing values.
* Engineered new features:
  * `FamilySize`: `df["FamilySize"] = df["SibSp"] + df["Parch"] + 1`
  * `IsAlone`: `df["IsAlone"] = (df["FamilySize"] == 1).astype(int)`

### Exploration
Used Pandas methods to inspect structure and statistical distributions:
* `df.info()`
* `df.describe()`
* `df["Survived"].value_counts()`
* `df.groupby()`

Compared survival rates across demographic baseline metrics like `Sex` and `Pclass`.

### Visualization
Used Matplotlib and Seaborn to generate:
* Survival by gender comparison.
* Survival rate by passenger class.
* Age distribution profiles.
* Fare vs. survival relationships.
* Confusion matrix heatmap for visual error analysis.

### Modeling
* Separated `Survived` as the target variable ($y$) and prepared features ($X$) for modeling.
* Applied an **80/20 train-test split** with `stratify=y` to maintain class distribution consistency.
* Trained a **Logistic Regression** model suitable for binary classification.

---

## 4. Key Findings

The exploration demonstrated that gender and passenger class were the primary drivers of survival rates:

* **Female passengers** had a survival rate of approximately **74.2%**.
* **Male passengers** had a survival rate of approximately **18.9%**.
* **1st-class passengers** had a survival rate of approximately **63.0%**.
* **2nd-class passengers** had a survival rate of approximately **47.3%**.
* **3rd-class passengers** had a survival rate of approximately **24.2%**.

### Highlight Insight
* **Key Finding:** Third-class passengers accounted for the majority of non-survivors, and passengers traveling with small families (2–4 members) had higher survival rates than those traveling alone or in large families.

---

## 5. Model Results

### Accuracy
* **Model Accuracy:** **84.36%**

The Logistic Regression model correctly classified the outcome for approximately 84 out of every 100 passengers in the test set.

### Confusion Matrix

| | Predicted: Did Not Survive | Predicted: Survived |
| :--- | :---: | :---: |
| **Actual: Did Not Survive** | **99** | 11 |
| **Actual: Survived** | 17 | **52** |

#### Interpretation
* **True Negatives:** 99 passengers who did not survive were correctly identified.
* **True Positives:** 52 passengers who survived were correctly identified.
* **False Positives:** 11 passengers were predicted to survive but did not.
* **False Negatives:** 17 passengers were predicted not to survive but survived.

---

## 6. Technologies Used

* **Python** — Core programming language
* **Pandas** — Data manipulation and analysis
* **NumPy** — Vectorized operations
* **Matplotlib** — Graphic plotting library
* **Seaborn** — Statistical visualization library
* **Scikit-learn** — Preprocessing, modeling, and metrics evaluation
* **Jupyter Notebook** — Interactive code execution environment

---

## 7. How to Run This Project

### Prerequisites
Install Python alongside required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

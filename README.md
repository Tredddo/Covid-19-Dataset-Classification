# Machine Learning project

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Environment](https://img.shields.io/badge/Google-Colab-yellow.svg)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-GPL_2.0-green.svg)](LICENSE)

---

## Disclaimer

> **DISCLAIMER:** The findings, models, and analytical pipelines presented in this repository are developed strictly for academic, educational, and scientific evaluation purposes. If the underlying dataset or derivative metrics pertain to clinical, diagnostic, or epidemiological variables, **this research must not be construed as clinical or medical advice**. The predictions and statistical inferences should under no circumstances be utilized for clinical diagnosis, therapeutic decisions, or real-world policy.

---

## Project Overview

This repository contains an academic research designed to explore, evaluate, and compare supervised machine learning architectures against an unsupervised baseline on structured **Categorical Ordinal** data. 

The primary research objective is to assess the generalization capabilities of three distinct classification algorithms:
1. **Decision Trees**
2. **Random Forests**
3. **Support Vector Machines (SVM)**

In addition to supervised classification, the pipeline integrates unsupervised exploratory stages with **Principal Component Analysis (PCA)** for projections and **K-Means Clustering** for cluster labels.

---

## Academic Context & Assignment

This study was undertaken as the capstone laboratory project for the **Machine Learning** course at **UniPG**.

**Consegna originale del docente (Italian):**

```text
Progetto Gruppo 1: COVID-19 Classification

Ciascun membro del gruppo utilizzerà un classificatore diverso tra questi:
-Decision Tree
-Random Forest
-SVM

Il progetto dovrà essere realizzato attenendosi al seguente schema:
1- Analisi preliminare (analisi dataset, formalizzazione del problema)
2- Preparazione dati (Data Preprocessing, Training/Test set split)
3- Addestramento dei classificatori (Training)
4- Valutazione dei risultati ( accuracy, recall, precision, F1, confusion matrix)
5- Confronto comune tra i tre algoritmi e i loro risultati
```

**English translation:**

```text
Group 1 Project: COVID-19 Classification

Each group member will use a different classifier from the following:
- Decision Tree
- Random Forest
- SVM

The project must be carried out according to the following outline:
1- Preliminary analysis (dataset analysis, problem formulation)
2- Data preparation (data preprocessing, training/test set split)
3- Training the classifiers
4- Evaluation of results (accuracy, recall, precision, F1 score, confusion matrix)
5- Comparative analysis of the three algorithms and their results
```

> [!NOTE]
> *The evaluation methodology mandates standardized 70/30 train-test partitioning, explicit seed (775) initialization for scientific reproducibility, hyperparameter tuning, and a rigorous theoretical-mathematical justification for every evaluation metric employed (Accuracy, Recall, Precision, F1-Score and confusion matrix)."*

### Applied methodology

The conceptual structure and exploratory sequence in this repository are based on the assignment and methodological guidelines established by our **professors**. While all source code implementations, experimental evaluations, and analytical deductions were independently authored by the project contributors.

---

## Project Pipeline & Documentation

The study is structured into 5 phases. Detailed logs, figures, and code explanations are documented in the `/docs` directory:

### Detailed Stage Breakdown & Module Links

| Syllabus Phase | Focus & Implementation Details | Technical Documentation |
| :--- | :--- | :--- |
| **01. Analisi preliminare** *(Preliminary analysis & Unsupervised exploration)* | Dataset profiling and formal problem definition. **Principal Component Analysis (PCA):** 2D latent projection. Unsupervised Phenotyping (**K-Means**): Identified natural clinical clusters from objective symptoms. The emergent clusters revealed distinct symptomatic severity tiers, allowing us to formalize and treat the target as **Categorical Ordinal** rather than relying on arbitrary labels. | [Phase 1: Preliminary Analysis & Unsupervised Topology](docs/01_preliminary_analysis.md) |
| **02. Preparazione dati** *(Data preprocessing & Train/Test set split)* | Target re-definition, we chose the discovered **K-Means clusters** as the main targets. **Feature pruning & Exclusion:** **Country:** (no biological relation to viral mechanisms) and **Severity** (Original target Discarded to prevent label contamination and ensure predictions rely strictly on objective symptom patterns). **Partitioning:** Stratified **70% Training / 30% Testing** split anchored to a deterministic seed (Random_state = 775). | [Phase 2: Data Preprocessing & Partitioning](docs/02_data_preparation.md) |
| **03. Addestramento dei classificatori** *(Classifiers training)* | Breakdown of all training phases and workflows to create **Decision tree, Random forest and Support Vector Machine (SVM)**. Commented step-by-step alongside executable Python/Colab blocks of code. | [Phase 3: Classifier Training & Optimization](docs/03_model_training.md) |
| **04. Valutazione dei risultati** *(Evaluation of results)* |  | [Phase 4: Validation & Confusion Matrices](docs/04_results_evaluation.md) |
| **05. Confronto comune tra i tre algoritmi** *(Comparative analysis)* | | [Phase 5: Comparative Benchmark & Final Synthesis](docs/05_comparative_benchmark.md) |

---

## Tech stack, tools and dataset

- **Runtime:** [Python 3.10+](https://www.python.org)
- **Google Colab Notebook:** [Google Colaboratory](https://colab.research.google.com)
- **Scientific Computing & Linear Algebra:** [NumPy](https://numpy.org)
- **Data Frame Manipulation:** [Pandas](https://pandas.pydata.org)
- **Machine Learning & Pipeline Architecture:** [scikit-learn](https://scikit-learn.org/stable)
- **Data Visualization:** [Matplotlib](https://matplotlib.org) & [Seaborn](https://seaborn.pydata.org)
- **Dataset (COVID-19 Symptoms Checker):** [kaggle-dataset](https://www.kaggle.com/datasets/iamhungundji/covid19-symptoms-checker/data)

---

## Authors

- [AJ](https://github.com/STORMJova) – [Decision Tree](colab/DecisionTree.py)
- [MM](https://github.com/Matteogit05) – [Random Forests](colab/RandomForest.py)
- [TB](https://github.com/Tredddo) – [Support Vector Machines](colab/SupportVectorMachines.py)

**Institution:** UniPG - AA 2025/2026

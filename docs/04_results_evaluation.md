# Phase 4: Validation & Confusion Matrices

> This document presents the evaluation of the three supervised classifiers — **Decision Tree**, **Random Forest**, and **Support Vector Machine (SVM)** — on the **unseen 30% test set** (81,000 patients). It defines the evaluation metrics, reports the confusion matrices, and provides a per-class error analysis. All models were trained in Phase 3 using the same 70/30 split (`random_state=775`) and evaluated on the same holdout set.

---

## 1. Evaluation Protocol

| Property | Value |
| :--- | :--- |
| Test set size | **81,000 samples** (30% of the cleaned cohort) |
| Training set size | 189,000 samples |
| Target | `Phenotype_Cluster` (4 nominal classes: 0, 1, 2, 3) |
| Features | 9 binary symptoms |
| `random_state` | **775** (identical split for all classifiers) |
| Metrics | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |

Because the task is **multiclass** (4 classes), precision, recall and F1 are computed using a **one-vs-rest** strategy: each class is treated as the positive class while the others are grouped as negative. The report provides both **macro** (unweighted average across classes) and **weighted** (average weighted by class support) aggregates.

---

## 2. Metric Definitions

### 2.1 Confusion Matrix

For a binary problem, the confusion matrix is:

| | Predicted Positive | Predicted Negative |
| :--- | :---: | :---: |
| **Actual Positive** | TP | FN |
| **Actual Negative** | FP | TN |

In the multiclass case, the matrix is extended to a 4×4 grid where entry `(i, j)` is the number of samples with true class `i` predicted as class `j`. The diagonal contains correct predictions; off-diagonal entries are errors.

### 2.2 Accuracy

Overall correctness across all classes:

$$
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
$$

For multiclass, it is simply the sum of the diagonal divided by the total number of samples.

### 2.3 Precision

Measures how many of the samples predicted as positive are actually positive:

$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$

High precision means few false positives.

### 2.4 Recall (Sensitivity)

Measures how many of the actual positive samples are correctly identified:

$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

High recall means few false negatives.

### 2.5 F1-Score

Harmonic mean of precision and recall, balancing the two:

$$
\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

Useful when precision and recall are in trade-off.

---

## 3. Decision Tree Evaluation

### 3.1 Classification Report

```
--- Decision Tree Classification Report ---

              precision    recall  f1-score   support

      Area 0       0.89      0.82      0.85     20566
      Area 1       0.62      0.82      0.70     21116
      Area 2       0.73      0.88      0.80     23055
      Area 3       1.00      0.37      0.54     16263

    accuracy                           0.75     81000
   macro avg       0.81      0.72      0.72     81000
weighted avg       0.79      0.75      0.73     81000
```

### 3.2 Confusion Matrix

| True \ Predicted | Area 0 | Area 1 | Area 2 | Area 3 |
| :---: | :---: | :---: | :---: | :---: |
| **Area 0** | **16,771** | 520 | 3,275 | 0 |
| **Area 1** | 0 | **17,343** | 3,773 | 0 |
| **Area 2** | 0 | 2,693 | **20,362** | 0 |
| **Area 3** | 2,149 | 7,638 | 517 | **5,959** |

### 3.3 Per-Class Metric Calculation (Area 0 Example)

For **Area 0**:

- **TP** = 16,771 (diagonal)
- **FP** = column 0 sum − TP = (0 + 0 + 2,149) = **2,149**
- **FN** = row 0 sum − TP = (520 + 3,275 + 0) = **3,795**
- **TN** = total − TP − FP − FN = 81,000 − 16,771 − 2,149 − 3,795 = **58,285**

Then:

$$
\text{Precision} = \frac{16{,}771}{16{,}771 + 2{,}149} = \frac{16{,}771}{18{,}920} \approx 0.89
$$

$$
\text{Recall} = \frac{16{,}771}{16{,}771 + 3{,}795} = \frac{16{,}771}{20{,}566} \approx 0.82
$$

$$
\text{F1} = 2 \cdot \frac{0.89 \cdot 0.82}{0.89 + 0.82} \approx 0.85
$$

$$
\text{Accuracy} = \frac{16{,}771 + 17{,}343 + 20{,}362 + 5{,}959}{81{,}000} = \frac{60{,}435}{81{,}000} \approx 0.75
$$

### 3.4 Error Analysis

The confusion matrix is not perfectly diagonal. Errors are **not random** but concentrated between **adjacent clusters**:

- **Area 3** (mild/paucisymptomatic) has precision = 1.00 but recall = 0.37: many true Area 3 patients are misclassified as Area 1 (7,638) or Area 0 (2,149). The tree never predicts Area 3 for those cases, so it avoids false positives but misses most true Area 3 instances.
- **Area 1** and **Area 2** are frequently confused with each other (3,773 and 2,693 errors respectively).
- **Area 0** is relatively well identified (recall 0.82) but loses some samples to Area 1 and Area 2.

These errors stem from the **geometric limitation** of a single decision tree: it can only make **axis-aligned splits** (horizontal or vertical cuts). The K-Means boundaries between phenotypes are **linear but diagonal** in the (`Total_Respiratory`, `Total_Systemic`) plane. A depth-4 tree cannot approximate a diagonal boundary without creating a staircase of many small orthogonal cuts, which it cannot do within the depth constraint. As a result, it misclassifies points near the diagonal boundaries.

---

## 4. Random Forest Evaluation

### 4.1 Classification Report

```
--- Random Forest Classification Report ---

              precision    recall  f1-score   support

           0       1.00      1.00      1.00     20566
           1       1.00      1.00      1.00     21116
           2       1.00      1.00      1.00     23055
           3       1.00      1.00      1.00     16263

    accuracy                           1.00     81000
   macro avg       1.00      1.00      1.00     81000
weighted avg       1.00      1.00      1.00     81000
```

### 4.2 Confusion Matrix

| True \ Predicted | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 |
| :---: | :---: | :---: | :---: | :---: |
| **Cluster 0** | **20,566** | 0 | 0 | 0 |
| **Cluster 1** | 0 | **21,116** | 0 | 0 |
| **Cluster 2** | 0 | 0 | **23,055** | 0 |
| **Cluster 3** | 0 | 0 | 0 | **16,263** |

### 4.3 Interpretation

The Random Forest achieves a **perfectly diagonal** confusion matrix: every one of the 81,000 test patients is classified correctly. This is possible because the ensemble aggregates **6 decision trees**, each trained on a bootstrap sample with a random subset of features. While a single tree produces a jagged, stepwise approximation of the diagonal K-Means boundaries, the **majority vote** across multiple trees effectively smooths these steps and reconstructs the linear boundaries with high fidelity. The result confirms the theoretical advantage of ensemble methods over a single tree when the decision boundary is not axis-aligned.

---

## 5. Support Vector Machine Evaluation

### 5.1 Classification Report

```
--- SVM Classification Report ---

              precision    recall  f1-score   support

           0       1.00      1.00      1.00     20566
           1       1.00      1.00      1.00     21116
           2       1.00      1.00      1.00     23055
           3       1.00      1.00      1.00     16263

    accuracy                           1.00     81000
   macro avg       1.00      1.00      1.00     81000
weighted avg       1.00      1.00      1.00     81000
```

### 5.2 Confusion Matrix

| True \ Predicted | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 |
| :---: | :---: | :---: | :---: | :---: |
| **Cluster 0** | **20,566** | 0 | 0 | 0 |
| **Cluster 1** | 0 | **21,116** | 0 | 0 |
| **Cluster 2** | 0 | 0 | **23,055** | 0 |
| **Cluster 3** | 0 | 0 | 0 | **16,263** |

### 5.3 Interpretation

The linear SVM also achieves a **perfectly diagonal** confusion matrix. This is not surprising: K-Means partitions the feature space into **Voronoi cells** whose boundaries are **linear hyperplanes** (in 2D, straight lines). A linear SVM searches for exactly such hyperplanes, maximizing the margin between classes. There is a **structural coincidence** between the geometry of the target (generated by K-Means) and the geometry of the classifier (linear kernel SVM). Consequently, the SVM reproduces the K-Means assignment rules exactly, without a single error on the test set.

---

## 6. Comparative Summary

| Classifier | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree** | 0.75 | 0.81 | 0.72 | 0.72 | 0.73 |
| **Random Forest** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| **SVM (linear)** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |

The Decision Tree is the only model that makes errors, with an accuracy of 0.75 and a macro F1 of 0.72. Its main weakness is the low recall on Area 3 (0.37) and the confusion between adjacent clusters. Random Forest and SVM both achieve perfect scores, but for different reasons:

- **Random Forest** approximates the diagonal boundaries through the ensemble of orthogonal trees.
- **SVM** matches the linear geometry of the K-Means target exactly.

---

## 7. Methodological Note on Perfect Scores

The perfect scores of Random Forest and SVM must be interpreted with caution. As already noted in Phase 1, the target `Phenotype_Cluster` is a **deterministic function** of the same symptoms used as input features. K-Means created the clusters by partitioning the (`Total_Respiratory`, `Total_Systemic`) space, which is itself a linear combination of the 9 binary symptoms. Therefore, the classification task is essentially to **reconstruct the K-Means boundaries** from the original symptoms.

High accuracy here measures how well each model **reproduces the K-Means assignment rules**, not how well it predicts clinical severity. The Decision Tree’s errors are purely geometric: it cannot draw diagonal lines. The Random Forest and SVM succeed because they can. This is a valuable lesson about the interplay between **data generation process** and **model geometry**, but it does not imply that these models would generalize to a real clinical prediction task with independent labels.

---

## 8. Phase 4 Deliverables

1. **Classification reports** for Decision Tree, Random Forest, and SVM on the 81,000-sample test set.
2. **Confusion matrices** for all three models, with per-class error analysis.
3. **Quantitative comparison** of Accuracy, Precision, Recall, and F1-Score (macro and weighted).
4. **Error analysis** for the Decision Tree, identifying the geometric limitations that cause misclassifications.
5. **Methodological caveat** on the perfect scores of Random Forest and SVM, explaining the deterministic relationship between features and target.
6. **Transition to Phase 5:** comparative benchmark and final synthesis of the three algorithms.

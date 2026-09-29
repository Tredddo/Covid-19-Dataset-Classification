# Phase 3: Classifier Training & Model Configuration

> This document describes the training phase of the three supervised classifiers — **Decision Tree**, **Random Forest**, and **Support Vector Machine (SVM)** — on the target `Phenotype_Cluster` produced by K-Means in Phase 1. All models share the same 70/30 split (`random_state=775`) defined in Phase 2, ensuring a fair and reproducible comparison.

---

## 1. Common Training Setup

All three classifiers are trained on the **9 original binary symptoms**, which constitute the feature matrix `X`, while the target `y` is the K-Means phenotype label:

```python
features = ['Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing',
            'Sore-Throat', 'Pains', 'Nasal-Congestion', 'Runny-Nose', 'Diarrhea']
# = respiratory_cols + systemic_cols

X = df_clean[features]
y = df_clean['Phenotype_Cluster']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=775
)
```

| Property | Value |
| :--- | :--- |
| Train / Test | 70% / 30% |
| `random_state` | 775 |
| Train size | 189,000 samples |
| Test size | 81,000 samples (unseen) |
| Features | 9 binary symptoms |
| Target | 4 nominal classes (`Phenotype_Cluster`) |

The derived columns (`Total_Respiratory`, `Total_Systemic`) are **excluded** from `X` to avoid feeding the classifiers the exact quantities used by K-Means to build the target, keeping the experiment methodologically clean.

---

## 2. Decision Tree

### 2.1 Rationale

The Decision Tree is a **white-box** classifier: it recursively partitions the feature space through axis-aligned, threshold-based splits, producing a set of if-then rules that can be inspected directly. It is scale-invariant, so no feature standardization is required.

### 2.2 Hyperparameters

| Parameter | Value | Motivation |
| :--- | :--- | :--- |
| `max_depth` | 4 | Limits tree complexity and prevents overfitting; keeps the tree interpretable. |
| `random_state` | 100 | Reproducibility of the split search. |

### 2.3 Training Code

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# X = binary symptoms (features)
# y = K-Means cluster (target)
features = ['Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing',
            'Sore-Throat', 'Pains', 'Nasal-Congestion', 'Runny-Nose', 'Diarrhea']
X = df_clean[features]
y = df_clean['Phenotype_Cluster']

# 70/30 split to test the tree on "unseen" patients
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=775
)

# Training
clf = DecisionTreeClassifier(max_depth=4, random_state=100)
clf.fit(X_train, y_train)

# Prediction on the test set
y_pred = clf.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[f'Area {i}' for i in range(4)])
plt.figure(figsize=(10, 8))
disp.plot(cmap='Blues', values_format='d')
plt.title('Decision Tree')
plt.grid(False)
plt.show()
```

### 2.4 Feature Importance (Gini)

The trained tree ranks the symptoms by their contribution to the impurity reduction:

| Symptom | Importance |
| :--- | :--- |
| Tiredness | 0.259 |
| Difficulty-in-Breathing | 0.256 |
| Fever | 0.147 |
| Runny-Nose | 0.142 |
| Pains | 0.094 |
| Diarrhea | 0.053 |
| Dry-Cough | 0.039 |
| Nasal-Congestion | 0.009 |
| Sore-Throat | 0.000 |

`Tiredness` and `Difficulty-in-Breathing` alone explain more than half of the splits, confirming that the systemic and respiratory axes identified in Phase 1 are the dominant discriminants.

### 2.5 Tree Visualization

```python
from sklearn.tree import plot_tree

plt.figure(figsize=(80, 10))
plot_tree(
    clf,
    feature_names=features,
    class_names=[f'Area {i}' for i in range(4)],
    filled=True,
    rounded=True,
    fontsize=12
)
plt.title("Decision Tree: Rules for Phenotype Classification")
plt.savefig('decision_tree_fenotipi.png')
plt.show()
```

### 2.6 Test-Set Performance

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

---

## 3. Random Forest

### 3.1 Rationale

The Random Forest is an **ensemble of decorrelated decision trees**. Each tree is trained on a bootstrap sample of the data and, at every split, only a random subset of features is considered. The final prediction is obtained by **majority voting** across all trees. This "wisdom of the crowd" mechanism smooths the piecewise-constant boundaries of a single tree and drastically reduces variance.

### 3.2 Hyperparameters

| Parameter | Value | Motivation |
| :--- | :--- | :--- |
| `n_estimators` | 6 | Small forest, sufficient to showcase the ensemble effect while keeping training fast. |
| `random_state` | 775 | Reproducibility across runs. |

No standardization is applied, since the base learners are threshold-based and therefore scale-invariant.

### 3.3 Training Code

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Original symptoms as X, to observe how they influence the clusters
# Exclude the columns we engineered (Total_Respiratory, Total_Systemic, Phenotype_Cluster)
features_originali = respiratory_cols + systemic_cols
X_rf = df_clean[features_originali]
y_rf = df_clean['Phenotype_Cluster']

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X_rf, y_rf, test_size=0.3, random_state=775
)

# Model training with random_state 775
rf = RandomForestClassifier(n_estimators=6, random_state=775)
rf.fit(X_train, y_train)

# Predictions:
# Each row of X_test traverses all n_estimators trees of the forest.
# The final class in y_pred is determined by the majority vote of the individual trees.
y_pred = rf.predict(X_test)

# Final report
print("--- Random Forest Classification Report ---\n")
print(classification_report(y_test, y_pred))
```

### 3.4 Test-Set Performance

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

### 3.5 Confusion Matrix

```python
from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[f'Cluster {i}' for i in range(4)])

plt.figure(figsize=(10, 8))
disp.plot(cmap='Greens', values_format='d', ax=plt.gca())
plt.title('Random Forest')
plt.grid(False)
plt.show()
```

The matrix is perfectly diagonal: the forest reconstructs the K-Means assignment rules exactly on the 81,000 unseen patients.

---

## 4. Support Vector Machine (SVM)

### 4.1 Rationale

The SVM searches for the **optimal separating hyperplane** that maximizes the margin between classes. With a **linear kernel**, it produces exactly the kind of geometric decision boundaries that K-Means implicitly defines (Voronoi-like partitions around centroids). Because the algorithm is distance- and margin-based, **feature standardization is mandatory**: the scaler is fitted on the training set only and then applied to the test set, preventing any information leakage.

### 4.2 Hyperparameters

| Parameter | Value | Motivation |
| :--- | :--- | :--- |
| `kernel` | `'linear'` | Matches the linearly separable structure of the K-Means target. |
| `gamma` | `'scale'` | Default heuristic, appropriate for standardized features. |
| `random_state` | 775 | Reproducibility. |

### 4.3 Training Code

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Use the features created earlier
features_originali = respiratory_cols + systemic_cols
X = df_clean[features_originali]
y = df_clean['Phenotype_Cluster']

# Data split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=775
)

# Homogenize the data through scaling
scaler_svm = StandardScaler()
X_train_scaled = scaler_svm.fit_transform(X_train)
X_test_scaled = scaler_svm.transform(X_test)

# Linear kernel
svm_model = SVC(kernel='linear', gamma='scale', random_state=775)
svm_model.fit(X_train_scaled, y_train)

# Report
y_pred_svm = svm_model.predict(X_test_scaled)

print("--- SVM Classification Report ---\n")
print(classification_report(y_test, y_pred_svm))
```

### 4.4 Test-Set Performance

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

### 4.5 Confusion Matrix

```python
from sklearn.metrics import ConfusionMatrixDisplay

cm_svm = confusion_matrix(y_test, y_pred_svm)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm,
                                  display_labels=[f'Cluster {i}' for i in range(4)])

plt.figure(figsize=(10, 8))
disp_svm.plot(cmap='Reds', values_format='d', ax=plt.gca())
plt.title('SVM')
plt.grid(False)
plt.show()
```

The linear SVM achieves a perfectly diagonal confusion matrix, isolating each K-Means cluster without a single misclassification.

---

## 5. Summary of Training Results

| Classifier | Scaling | Key Hyperparameters | Test Accuracy | Macro F1 |
| :--- | :---: | :--- | :---: | :---: |
| Decision Tree | No | `max_depth=4`, `random_state=100` | 0.75 | 0.72 |
| Random Forest | No | `n_estimators=6`, `random_state=775` | **1.00** | **1.00** |
| SVM | Yes (`StandardScaler`) | `kernel='linear'`, `random_state=775` | **1.00** | **1.00** |

---

## 6. Phase 3 Deliverables

1. **Decision Tree** trained with `max_depth=4` and `random_state=100`, with feature importances and a white-box tree visualization.
2. **Random Forest** trained with `n_estimators=6` and `random_state=775`, evaluated on the 81,000-sample test set.
3. **Linear SVM** trained on standardized features (`fit_transform` on train, `transform` on test), with `random_state=775`.
4. **Standardized evaluation artifacts** (classification reports and confusion matrices) for all three models, ready for the comparative analysis in Phase 5.
5. **Transition to Phase 4:** quantitative validation and error analysis of the three classifiers on the unseen test set.

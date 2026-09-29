# Phase 2: Data Preprocessing & Train/Test Partitioning

> This document describes how the dataset is cleaned, how the target is defined and how the data is split before training the three classifiers (Decision Tree, Random Forest, SVM).

---

## 1. Data Cleaning

Only variables relevant to the clinical analysis are kept; non-functional data is excluded.

| Excluded variable | Motivation |
| :--- | :--- |
| `Country` | Non-biological confounder, irrelevant to the virus dynamics. |
| `Severity_*` (original target) | Excluded to obtain an analysis based purely on objective symptomatology and to avoid label leakage. The K-Means step generates new targets ignoring pre-existing labels. |

**Row filtering.** Records with no real symptoms are removed (`None_Sympton == 1`, `None_Experiencing == 1`, or `Total_Respiratory == 0` and `Total_Systemic == 0`).

- Original shape: **316,800 rows**
- After filtering: **270,000 rows**

---

## 2. Target Definition (K-Means)

Symptoms are aggregated into two macro-categories:

- **Respiratory:** `Dry-Cough`, `Difficulty-in-Breathing`, `Sore-Throat`, `Nasal-Congestion`, `Runny-Nose`
- **Systemic:** `Fever`, `Tiredness`, `Pains`, `Diarrhea`

```python
df['Total_Respiratory'] = df[respiratory_cols].sum(axis=1)
df['Total_Systemic']    = df[systemic_cols].sum(axis=1)

X = df_clean[['Total_Respiratory', 'Total_Systemic']]
X_scaled = StandardScaler().fit_transform(X)   # needed for Euclidean distances

kmeans = KMeans(n_clusters=4, random_state=775, n_init=300)
df_clean['Phenotype_Cluster'] = kmeans.fit_predict(X_scaled)
```

Resulting cluster sizes: 68,400 / 70,200 / 77,400 / 54,000 (total 270,000).
`Phenotype_Cluster` is the target `y` for all classifiers.

---

## 3. Feature Matrix

The input `X` consists of the **9 original binary symptoms**:

```python
features = ['Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing',
            'Sore-Throat', 'Pains', 'Nasal-Congestion', 'Runny-Nose', 'Diarrhea']
# = respiratory_cols + systemic_cols
X = df_clean[features]
y = df_clean['Phenotype_Cluster']
```

The derived columns (`Total_Respiratory`, `Total_Systemic`) are **not** used as features.

---

## 4. Training/Test Split

| Parameter | Value |
| :--- | :--- |
| Train / Test | **70% / 30%** (same split used during the lectures) |
| `random_state` | **775** (guarantees reproducibility) |
| Train size | 189,000 samples |
| Test size | **81,000 samples** (unseen during training) |

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=775
)
```

The split is random and identical for all three classifiers. Test-set supports (20,566 / 21,116 / 23,055 / 16,263) match the global cluster proportions, so classes are well represented.

---

## 5. Model-Specific Preprocessing

| Classifier | Scaling | Notes |
| :--- | :--- | :--- |
| Decision Tree | No | Threshold-based splits are scale-invariant. `max_depth=4`, `random_state=100`. |
| Random Forest | No | Same reason. `n_estimators=6`, `random_state=775`. |
| SVM | **Yes** (`StandardScaler`) | Distance/margin-based: scaler is fitted on the training set only and applied to the test set (`fit_transform` on train, `transform` on test). Linear kernel, `random_state=775`. |

---

## 6. Phase 2 Deliverables

1. Cleaned cohort of 270,000 symptomatic patients, without `Country` and `Severity_*`.
2. Target `y = Phenotype_Cluster` (4 classes) and feature matrix `X` of 9 binary symptoms.
3. Reproducible 70/30 partition (`random_state=775`): 189,000 train / 81,000 test.
4. Transition to Phase 3: classifier training.

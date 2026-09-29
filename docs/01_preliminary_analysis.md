# Phase 1: Preliminary Analysis & Problem Formalization

> This document details the initial phase of the machine learning pipeline: raw dataset profiling, clinical feature taxonomy, dimensional exploration via **Principal Component Analysis (PCA)** and data-driven clinical phenotyping through **K-Means Clustering**.

---

## 1. Raw Dataset Profile & Architecture

The analysis is conducted on `Cleaned-Data.csv`, originally collected from the Kaggle *COVID-19 Symptoms Checker* repository.

### 1.1 Dimensionality & Attribute Encoding

- **Sample Space:** 316,800 patient records (one row per patient).
- **Feature Space:** 27 attributes. Except for the nominal string variable `Country`, all features are binary-encoded [0, 1].

### 1.2 Clinical Taxonomy & Feature Grouping

1. **Symptomatology:**
   - *Respiratory:* `Dry-Cough`, `Difficulty-in-Breathing`, `Sore-Throat`, `Nasal-Congestion`, `Runny-Nose`.
   - *Systemic / General:* `Fever`, `Tiredness`, `Pains`, `Diarrhea`.
   - *Absence indicators:* `None_Sympton`, `None_Experiencing`.
2. **Demographics:**
   - *Age:* `Age_0-9`, `Age_10-19`, `Age_20-24`, `Age_25-59`, `Age_60+`.
   - *Gender:* `Gender_Female`, `Gender_Male`, `Gender_Transgender`.
3. **Epidemiological Context:** `Contact_Yes`, `Contact_No`, `Contact_Dont-Know`.
4. **Original Target & Origin:**
   - *Original severity labels:* `Severity_None`, `Severity_Mild`, `Severity_Moderate`, `Severity_Severe` (one-hot, 79,200 samples each, balanced 25%).
   - *Geographical origin:* `Country`.

---

## 2. Problem Formulation

The research question is: **"How severe is a patient, given their symptom presentation?"**

The original labels follow a natural order (None < Mild < Moderate < Severe), which suggested an ordinal classification. However, as shown in Section 3, these labels are not supported by the symptoms. They are therefore discarded and replaced by a target derived from the data (Section 5).

The final task is a **4-class multiclass classification**: the K-Means phenotypes are **nominal** categories, with no intrinsic order.

---

## 3. Dimensional Exploration via PCA

A 2D **PCA** was performed to check whether a linear projection of the binary features could separate the original severity classes (`Target_Severity`).

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load data and consolidate one-hot severity target
df = pd.read_csv('Cleaned-Data.csv')
severity_cols = ['Severity_None', 'Severity_Mild', 'Severity_Moderate', 'Severity_Severe']
df['Target_Severity'] = df[severity_cols].idxmax(axis=1)

# 2. Sample 10,000 instances for visual clarity
df_sample = df.sample(n=10000, random_state=775)
X_sample = df_sample.drop(columns=severity_cols + ['Country', 'Target_Severity'])
y_sample = df_sample['Target_Severity']

# 3. Standardize features and compute 2 principal components
X_scaled = StandardScaler().fit_transform(X_sample)
pca = PCA(n_components=2)
pca_res = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(data=pca_res, columns=['PCA1', 'PCA2'])
pca_df['Severity'] = y_sample.values

# 4. 2D scatter plot
plt.figure(figsize=(12, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue='Severity', data=pca_df, alpha=0.5, palette='viridis')
plt.title('Scatter Plot del Database tramite PCA (Riduzione Dimensionale)')
plt.xlabel('Componente Principale 1')
plt.ylabel('Componente Principale 2')
plt.grid(True, alpha=0.3)
plt.savefig('docs/images/scatter_pca_covid.png')
plt.show()
```

### PCA Findings & Limitations

The projection of 10,000 records shows **complete visual overlap among the four original severity classes**. Two consequences:

1. The original labels do not correspond to separable symptom patterns.
2. The relation between symptoms and original severity is not captured by a linear 2D projection (this does not exclude non-linear or higher-dimensional structure).

---

## 4. Clinical Feature Re-Engineering: Respiratory & Systemic Axes

Since the original labels are unreliable, binary symptoms are aggregated into two clinical dimensions:

- **`Total_Respiratory`:** sum of `Dry-Cough`, `Difficulty-in-Breathing`, `Sore-Throat`, `Nasal-Congestion`, `Runny-Nose` (range 0-5).
- **`Total_Systemic`:** sum of `Fever`, `Tiredness`, `Pains`, `Diarrhea` (range 0-4).

Records with conflicting or null clinical information (`None_Sympton == 1`, `None_Experiencing == 1`, or both totals equal to 0) are removed, reducing the cohort from 316,800 to **270,000 symptomatic patients**.

---

## 5. Unsupervised Phenotyping via K-Means Clustering

**K-Means** is applied on the standardized bivariate space (`Total_Respiratory`, `Total_Systemic`) to identify data-driven phenotypes.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

respiratory_cols = ['Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat', 'Nasal-Congestion', 'Runny-Nose']
systemic_cols = ['Fever', 'Tiredness', 'Pains', 'Diarrhea']

df['Total_Respiratory'] = df[respiratory_cols].sum(axis=1)
df['Total_Systemic'] = df[systemic_cols].sum(axis=1)

condition = (
    (df['None_Sympton'] == 0) &
    (df['None_Experiencing'] == 0) &
    ((df['Total_Respiratory'] > 0) | (df['Total_Systemic'] > 0))
)
df_clean = df[condition].copy()

scaler = StandardScaler()
X_pheno = scaler.fit_transform(df_clean[['Total_Respiratory', 'Total_Systemic']])

kmeans = KMeans(n_clusters=4, random_state=775, n_init=300)
df_clean['Phenotype_Cluster'] = kmeans.fit_predict(X_pheno)
```

### 5.1 Centroids & Phenotype Distribution

| Cluster ID | Sample Count | Clinical Characterization |
| --- | --- | --- |
| **Cluster 0** | 68,400 | **Systemic Dominant:** low respiratory burden, high systemic burden. |
| **Cluster 1** | 70,200 | **Respiratory Dominant:** high respiratory burden (3-5), low systemic burden (0-1). |
| **Cluster 2** | 77,400 | **Combined:** high scores on both axes. |
| **Cluster 3** | 54,000 | **Mild / Paucisymptomatic:** low scores on both axes. |

> The cluster descriptions follow the scatter plot of the centroids. Cluster IDs depend on the run: verify them with `scaler.inverse_transform(kmeans.cluster_centers_)` before finalizing.

---

## 6. Summary of Phase 1 Deliverables

1. **Feature pruning:** `Country` and the original `Severity_*` columns are removed to avoid geographical confounding and target leakage.
2. **Target definition:** `Phenotype_Cluster` is the target of all downstream classifiers (nominal, 4 classes).
3. **Methodological note:** the target is a deterministic function of the same symptoms later used as input features. High classification scores are therefore expected and measure how well each model reproduces the K-Means boundaries, not clinical prediction.
4. **Transition to Phase 2:** the cleaned cohort (270,000 samples) and the 9 binary symptoms form the input for the random 70/30 train/test split (`random_state=775`).

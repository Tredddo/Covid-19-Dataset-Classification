# Phase 1: Preliminary Analysis & Problem Formalization

>This document details the initial phase of the machine learning pipeline, focusing on raw dataset profiling, clinical feature taxonomy, dimensional exploration via **Principal Component Analysis (PCA)**, and data-driven clinical phenotyping through **K-Means Clustering**.

---

## 1. Raw Dataset Profile & Architecture

The empirical analysis is conducted on `Cleaned-Data.csv`, originally collected from the Kaggle *COVID-19 Symptoms Checker* repository.

### 1.1 Dimensionality & Attribute Encoding

- **Sample Space:** The raw dataset contains 316,800 individual patient records.

- **Feature Space:** It comprises 27 recorded attributes. With the exception of the nominal string variable `Country` (residence nation), all remaining clinical and demographic features are binary-encoded [0, 1].

### 1.2 Clinical Taxonomy & Feature Grouping

To ensure medical interpretability, features are categorized into four distinct functional groups:

1. **Symptomatology:**
- *Respiratory Symptoms:* `Dry-Cough`, `Difficulty-in-Breathing`, `Sore-Throat`, `Nasal-Congestion`, `Runny-Nose`.
- *Systemic / General Symptoms:* `Fever`, `Tiredness`, `Pains`, `Diarrhea`.
- *Absence Indicators:* `None_Sympton`, `None_Experiencing`.

2. **Demographics:**
- *Age Strata:* `Age_0-9`, `Age_10-19`, `Age_20-24`, `Age_25-59`, `Age_60+`.
- *Gender:* `Gender_Female`, `Gender_Male`, `Gender_Transgender`.

3. **Epidemiological Context:**
- Risk exposure: `Contact_Yes`, `Contact_No`, `Contact_Dont-Know`.

4. **Pre-Existing Target & Origin:**
* *Original Severity Labels:* Four one-hot encoded columns: `Severity_None`, `Severity_Mild`, `Severity_Moderate`, `Severity_Severe` (79,200 samples each, forming a balanced 25% distribution across classes).
* *Geographical Origin:* `Country`.

---

## 2. Problem Formulation

The primary diagnostic objective is to address the following research question: **"What is the actual clinical severity level of a patient given their objective symptom presentation?"**

From a statistical learning perspective, clinical disease progression exhibits an intrinsic, monotonic hierarchy:
- None
- Mild
- Moderate
- Severe

Consequently, the task is formally defined as an **Ordinal Classification** problem.

---

## 3. Dimensional Exploration via PCA (Principal Component Analysis)

Prior to model training, a 2D **Principal Component Analysis (PCA)** was performed to evaluate whether linear orthogonal projections of binary symptoms were sufficient to separate the original four severity classes (`Target_Severity`) in Euclidean space.

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
df_sample = df.sample(n=10000, random_state=42)
X_sample = df_sample.drop(columns=severity_cols + ['Country', 'Target_Severity'])
y_sample = df_sample['Target_Severity']

# 3. Standardize features and compute 2 principal components
X_scaled = StandardScaler().fit_transform(X_sample)
pca = PCA(n_components=2)
pca_res = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(data=pca_res, columns=['PCA1', 'PCA2'])
pca_df['Severity'] = y_sample.values

# 4. Generate 2D Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Severity', data=pca_df, alpha=0.5, palette='viridis')
plt.title('Scatter Plot del Database tramite PCA (Riduzione Dimensionale)')
plt.xlabel('Componente Principale 1')
plt.ylabel('Componente Principale 2')
plt.grid(True, alpha=0.3)
plt.savefig('docs/images/scatter_pca_covid.png')
plt.show()

```

### PCA Findings & Methodological Limitations

The 2D projection over 10,000 patient records demonstrates **complete visual overlap among all four pre-existing severity classes** along the first two principal component axes.

This empirical outcome highlights two critical constraints:

1. The synthetic ground-truth labels in the original dataset exhibit high entropy and do not correspond to clear, separable symptom boundaries.
2. The mapping between discrete symptoms and clinical outcomes cannot be captured via linear dimensionality reduction.

---

## 4. Clinical Feature Re-Engineering: Respiratory & Systemic Axes

To overcome the inconsistency of raw labels, the methodology introduces a domain-guided re-engineering step by aggregating binary indicators into two clinical dimensions:

* **Respiratory Symptom Burden (`Total_Respiratory`):** The sum of local respiratory manifestations (`Dry-Cough`, `Difficulty-in-Breathing`, `Sore-Throat`, `Nasal-Congestion`, `Runny-Nose`).
* **Systemic Symptom Burden (`Total_Systemic`):** The sum of generalized systemic markers (`Fever`, `Tiredness`, `Pains`, `Diarrhea`).

To eliminate noise from asymptomatic entries, samples with conflicting or null clinical records (`None_Sympton == 1`, `None_Experiencing == 1`, or total symptom count equal to 0) were filtered out.

* This filtering reduced the active cohort from 316,800 to **270,000 symptomatic patient instances**.

---

## 5. Unsupervised Phenotyping via K-Means Clustering

Using the standardized bivariate space (`Total_Respiratory`, `Total_Systemic`), an unsupervised **K-Means** clustering algorithm was applied to identify data-driven clinical phenotypes.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Feature aggregation across physiological axes
respiratory_cols = ['Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat', 'Nasal-Congestion', 'Runny-Nose']
systemic_cols = ['Fever', 'Tiredness', 'Pains', 'Diarrhea']

df['Total_Respiratory'] = df[respiratory_cols].sum(axis=1)
df['Total_Systemic'] = df[systemic_cols].sum(axis=1)

# 2. Filter cohort to strictly symptomatic patient records
condition = (
    (df['None_Sympton'] == 0) & 
    (df['None_Experiencing'] == 0) & 
    ((df['Total_Respiratory'] > 0) | (df['Total_Systemic'] > 0))
)
df_clean = df[condition].copy()

# 3. Standardize dimensions for Euclidean distance calculation
scaler = StandardScaler()
X_pheno = scaler.fit_transform(df_clean[['Total_Respiratory', 'Total_Systemic']])

# 4. Partition into k=4 stable clinical clusters
kmeans = KMeans(n_clusters=4, random_state=775, n_init=300)
df_clean['Phenotype_Cluster'] = kmeans.fit_predict(X_pheno)

```

### 5.1 Centroids & Phenotype Distribution

The K-Means model converged onto 4 geometric clusters corresponding to distinct clinical profiles:

| Cluster ID | Sample Count | Clinical Characterization |
| --- | --- | --- |
| **Cluster 0** | 68,400 | **Systemic Dominant:** Low respiratory burden, elevated fever and myalgia. |
| **Cluster 1** | 70,200 | **Mild / Paucisymptomatic:** Low manifestation scores across both physiological axes. |
| **Cluster 2** | 77,400 | **Severe Combined:** High scores on both respiratory and systemic dimensions. |
| **Cluster 3** | 54,000 | **Respiratory Dominant:** High dyspnea and cough with minimal systemic involvement. |

---

## 6. Summary of Phase 1 Deliverables

1. **Feature Pruning:** `Country` and the original `Severity_*` targets are removed from the supervised training set to eliminate geographical confounding and target leakage.

2. **Ground-Truth Target Definition:** The discovered `Phenotype_Cluster` labels serve as the validated **Categorical Ordinal** target for all downstream classification models.

3. **Transition to Phase 2:** The cleaned cohort (270,000 samples) and the original individual binary symptoms form the input dataset for stratified train/test partitioning.

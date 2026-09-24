import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns

# Caricamento e preparazione dati
df = pd.read_csv('Cleaned-Data.csv')
severity_cols = ['Severity_None', 'Severity_Mild', 'Severity_Moderate', 'Severity_Severe']
df['Target_Severity'] = df[severity_cols].idxmax(axis=1)

# Selezione Feature e Campionamento
# Usiamo un campione di 10.000 righe per rendere il grafico leggibile
df_sample = df.sample(n=10000, random_state=775)
X_sample = df_sample.drop(columns=severity_cols + ['Country', 'Target_Severity'])
y_sample = df_sample['Target_Severity']

# Standardizzazione e PCA
X_scaled = StandardScaler().fit_transform(X_sample)
pca = PCA(n_components=2)
pca_res = pca.fit_transform(X_scaled)

# Creazione DataFrame per lo scatter
pca_df = pd.DataFrame(data=pca_res, columns=['PCA1', 'PCA2'])
pca_df['Severity'] = y_sample.values

# Plot
plt.figure(figsize=(12, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue='Severity', data=pca_df, alpha=0.5, palette='viridis')
plt.title('Scatter Plot del Database tramite PCA (Riduzione Dimensionale)')
plt.xlabel('Componente Principale 1')
plt.ylabel('Componente Principale 2')
plt.grid(True, alpha=0.3)
plt.savefig('scatter_pca_covid.png')
plt.show()

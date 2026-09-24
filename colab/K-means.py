import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Caricamento dataset
df = pd.read_csv('Cleaned-Data.csv')

# Definisco i gruppi di sintomi
respiratory_cols = ['Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat', 'Nasal-Congestion', 'Runny-Nose']
systemic_cols = ['Fever', 'Tiredness', 'Pains', 'Diarrhea']

# Creo le due nuove feature sommando i sintomi
df['Total_Respiratory'] = df[respiratory_cols].sum(axis=1) # Cambiamo l'asse
df['Total_Systemic'] = df[systemic_cols].sum(axis=1)

# Filtro: tengo solo i pazienti che hanno effettivamente dei sintomi
condition = (
    (df['None_Sympton'] == 0) &
    (df['None_Experiencing'] == 0) &
    ((df['Total_Respiratory'] > 0) | (df['Total_Systemic'] > 0))
)
df_clean = df[condition].copy()
# (Rimuovo chi ha 'None_Sympton' a 1 o somma sintomi a 0)

# Controllo veloce per vedere quante righe ho perso
print(f"Shape originale: {df.shape}")
print(f"Shape dopo filtro: {df_clean.shape}")

# Preparazione dati per KMeans
X = df_clean[['Total_Respiratory', 'Total_Systemic']]

# Standardizzo i dati (necessario per le distanze euclidee del KMeans)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering su 4 gruppi
kmeans = KMeans(n_clusters=4, random_state=775, n_init=300)
df_clean['Phenotype_Cluster'] = kmeans.fit_predict(X_scaled)

# Vediamo come sono distribuiti i cluster
print("\nDistribuzione Cluster:")
print(df_clean['Phenotype_Cluster'].value_counts().sort_index())


"""
    PLOT
"""

import matplotlib.pyplot as plt

# Recupero i centroidi convertiti (dai numeri scalati a quelli reali)
centroids_real = scaler.inverse_transform(kmeans.cluster_centers_)

plt.figure(figsize=(10, 8))

# Plot dei Pazienti
# I dati sono numeri interi e si sovrappongono.
scatter = plt.scatter(
    df_clean['Total_Respiratory'],
    df_clean['Total_Systemic'],
    c=df_clean['Phenotype_Cluster'],
    cmap='viridis',
    s=100, # Dimensione punti
    alpha=1 # Trasparenza fondamentale per vedere la densità
)

# Plot dei Centroidi
plt.scatter(
    centroids_real[:, 0],
    centroids_real[:, 1],
    c='red',
    marker='X',
    s=600,
    linewidths=2,
    label='Centroidi'
)
plt.title('Fenotipi')
plt.xlabel('Somma Sintomi Respiratori (Dry-Cough, Dyspnea, etc.)')
plt.ylabel('Somma Sintomi Sistemici (Fever, Tiredness, etc.)')
plt.grid(True, linestyle='--', alpha=0.3)

# Legenda automatica
plt.legend(*scatter.legend_elements(), loc="upper left", title="Cluster")
plt.show()

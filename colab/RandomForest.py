from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Usiamo i sintomi originali come X per vedere come influenzano i cluster
# Escludiamo le colonne che abbiamo creato noi (Total_Respiratory, Total_Systemic, Phenotype_Cluster)
features_originali = respiratory_cols + systemic_cols
X_rf = df_clean[features_originali]
y_rf = df_clean['Phenotype_Cluster']

# Split in training e test set
X_train, X_test, y_train, y_test = train_test_split(X_rf, y_rf, test_size=0.3, random_state=775)

# Allenamento del modello con random_state 775
rf = RandomForestClassifier(n_estimators=6, random_state=775)
rf.fit(X_train, y_train)

# Generazione delle predizioni:
# Ogni riga di X_test attraversa tutti gli n_estimators (alberi) della foresta.
# La classe finale in y_pred è determinata dalla maggioranza dei voti espressi dai singoli alberi
y_pred = rf.predict(X_test)

# Report Finale
print("--- Report di Classificazione RF ---\n")
print(classification_report(y_test, y_pred))
"""
--- Report di Classificazione RF ---

              precision    recall  f1-score   support

           0       1.00      1.00      1.00     20566
           1       1.00      1.00      1.00     21116
           2       1.00      1.00      1.00     23055
           3       1.00      1.00      1.00     16263

    accuracy                           1.00     81000
   macro avg       1.00      1.00      1.00     81000
weighted avg       1.00      1.00      1.00     81000
"""


"""
    Confusion matrix
"""

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Matrice di confusione
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f'Cluster {i}' for i in range(4)])

plt.figure(figsize=(10, 8))
disp.plot(cmap='Greens', values_format='d', ax=plt.gca())

plt.title('Random Forest')
plt.grid(False)
plt.show()

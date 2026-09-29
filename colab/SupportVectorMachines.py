from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Utilizziamo le feature create in precedenza
features_originali = respiratory_cols + systemic_cols
X = df_clean[features_originali]
y = df_clean['Phenotype_Cluster']

# Split dei dati
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=775)

# Rendiamo i dati omogenei con lo scaling
scaler_svm = StandardScaler()
X_train_scaled = scaler_svm.fit_transform(X_train)
X_test_scaled = scaler_svm.transform(X_test)

# Usiamo un kernel='linear'
svm_model = SVC(gamma='scale', random_state=775)
svm_model.fit(X_train_scaled, y_train)

# Report
y_pred_svm = svm_model.predict(X_test_scaled)

print("--- Report di Classificazione SVM ---\n")
print(classification_report(y_test, y_pred_svm))
"""
--- Report di Classificazione SVM ---

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

cm_svm = confusion_matrix(y_test, y_pred_svm)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=[f'Cluster {i}' for i in range(4)])

plt.figure(figsize=(10, 8))
disp_svm.plot(cmap='Reds', values_format='d', ax=plt.gca())

plt.title('SVM')
plt.grid(False)
plt.show()

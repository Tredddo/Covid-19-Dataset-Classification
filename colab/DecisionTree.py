import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# X = Sintomi binari (Features)
# Y = Cluster del K-Means (Target)
features = [
    'Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing',
    'Sore-Throat', 'Pains', 'Nasal-Congestion', 'Runny-Nose', 'Diarrhea'
]
X = df_clean[features]
y = df_clean['Phenotype_Cluster']

# Dividiamo i dati per testare l'albero su pazienti "mai visti"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=775)

# Addestramento
clf = DecisionTreeClassifier(max_depth=4, random_state=100)
clf.fit(X_train, y_train)


importances = pd.DataFrame({
    'Sintomo': features,
    'Importanza': clf.feature_importances_
}).sort_values(by='Importanza', ascending=False)

print(importances)
"""
                   Sintomo  Importanza
1                Tiredness    0.259293
3  Difficulty-in-Breathing    0.256483
0                    Fever    0.147390
7               Runny-Nose    0.141918
5                    Pains    0.093833
8                 Diarrhea    0.052763
2                Dry-Cough    0.038990
6         Nasal-Congestion    0.009330
4              Sore-Throat    0.000000
"""


"""
    Confusion matrix
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier

# X = Sintomi binari (Features)
# y = Cluster del K-Means (Target)
features = [
    'Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing',
    'Sore-Throat', 'Pains', 'Nasal-Congestion', 'Runny-Nose', 'Diarrhea'
]
X = df_clean[features]
y = df_clean['Phenotype_Cluster']

# Dividiamo i dati per testare l'albero su pazienti "mai visti"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=775)

# Addestramento
clf = DecisionTreeClassifier(max_depth=4, random_state=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Matrice di confusione
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f'Area {i}' for i in range(4)])
plt.figure(figsize=(10, 8))
disp.plot(cmap='Blues', values_format='d')
plt.title('Decision Tree')
plt.grid(False)
plt.show()


"""
    PLOT
"""

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Plot albero (white box)
plt.figure(figsize=(80, 10))
plot_tree(
    clf,
    feature_names=features,
    class_names=[f'Area {i}' for i in range(4)],
    filled=True,
    rounded=True,
    fontsize=12
)
plt.title("Albero di Decisione: Regole per la Classificazione dei Fenotipi")
plt.savefig('decision_tree_fenotipi.png')
plt.show()


"""
    REPORT
"""

from sklearn.metrics import classification_report

# Report Finale
print("--- Report di Classificazione DT ---\n")
print(classification_report(y_test, y_pred, target_names=[f'Area {i}' for i in range(4)]))
"""
--- Report di Classificazione DT ---

              precision    recall  f1-score   support

      Area 0       0.89      0.82      0.85     20566
      Area 1       0.62      0.82      0.70     21116
      Area 2       0.73      0.88      0.80     23055
      Area 3       1.00      0.37      0.54     16263

    accuracy                           0.75     81000
   macro avg       0.81      0.72      0.72     81000
weighted avg       0.79      0.75      0.73     81000
"""


"""
    Tree visualization
"""

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Plot albero (white box)
plt.figure(figsize=(80, 10))
plot_tree(
    clf,
    feature_names=features,
    class_names=[f'Area {i}' for i in range(4)],
    filled=True,
    rounded=True,
    fontsize=12
)
plt.title("Albero di Decisione: Regole per la Classificazione dei Fenotipi")
plt.savefig('decision_tree_fenotipi.png')
plt.show()

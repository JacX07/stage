import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Charger les données
df = pd.read_csv("diabetes.csv")
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 2. Séparer (exactement comme pour votre TRM)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Entraîner une Forêt Aléatoire (Random Forest)
modele = RandomForestClassifier(random_state=42)
modele.fit(X_train, y_train)

# 4. Évaluer
print(f"Accuracy Train : {accuracy_score(y_train, modele.predict(X_train)):.2%}")
print(f"Accuracy Test  : {accuracy_score(y_test, modele.predict(X_test)):.2%}")
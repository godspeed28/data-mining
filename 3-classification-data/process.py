import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset Besar
data = load_breast_cancer()

X = data.data        # 569 data, 30 fitur
y = data.target      # kelas (0 = malignant, 1 = benign)

df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y

print(df.shape)
df.head()

# Pisahkan Fitur & Target
X = df.drop('target', axis=1)
y = df['target']

# Split Data (Training & Testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalisasi Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Klasifikasi (Random Forest – cocok untuk data besar)
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi Model
print("Akurasi:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred))

# Feature Importance
importance = pd.DataFrame({
    'Fitur': data.feature_names,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(importance.head(10))





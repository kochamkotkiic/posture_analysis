#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# Wczytaj dane
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))))
CSV_PATH = os.path.join(BASE_DIR, "data", "posture_data.csv")

df = pd.read_csv(CSV_PATH)

print(f"Kształt danych: {df.shape}")
print(f"\nRozkład klas:")
print(df["label"].value_counts())
print(f"\nCzy są NULLe: {df.isnull().sum().sum()}")
df.head()


# In[8]:


# Oddziel cechy od etykiet
X = df.drop("label", axis=1)
y = df["label"]

# === NOWY KOD: NORMALIZACJA DANYCH TRENINGOWYCH ===
# 1. Wyliczamy "idealną pozycję" (bazę) dla nagrania, uśredniając wiersze, gdzie postawa była "Dobra" (label 0)
good_posture_rows = df[df["label"] == 0].drop("label", axis=1)
calibration_base = good_posture_rows.mean().to_dict()

# 2. Odejmujemy tę idealną bazę od KAŻDEGO wiersza w zbiorze danych
# Dzięki temu w tabeli X znajdą się tylko RÓŻNICE (czyli to samo, co daje normalizer.py)
for col in X.columns:
    X[col] = X[col] - calibration_base[col]

print("✅ Znormalizowano dane treningowe!")
# ==================================================

print(f"Liczba cech: {X.shape[1]}")
print(f"Nazwy cech: {list(X.columns)}")

# Podział na zbiór treningowy i testowy (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Skalowanie (ważne dla SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\nTreningowe: {X_train.shape}")
print(f"Testowe:    {X_test.shape}")


# In[9]:


rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
# === ZMIANA: używamy przeskalowanych danych (X_train_scaled zamiast X_train) ===
rf_model.fit(X_train_scaled, y_train)  

# === ZMIANA: testujemy na przeskalowanych danych ===
y_pred_rf = rf_model.predict(X_test_scaled)

print("=== RANDOM FOREST ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=["Dobra", "Zla"]))


# In[10]:


svm_model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True,
    random_state=42
)
svm_model.fit(X_train_scaled, y_train)  # SVM wymaga skalowania

y_pred_svm = svm_model.predict(X_test_scaled)

print("=== SVM ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm, target_names=["Dobra", "Zla"]))


# In[11]:


fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, (name, y_pred) in zip(axes, [
    ("Random Forest", y_pred_rf),
    ("SVM",           y_pred_svm)
]):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Dobra", "Zla"],
                yticklabels=["Dobra", "Zla"])
    ax.set_title(f"{name}\nAccuracy: {accuracy_score(y_test, y_pred):.3f}")
    ax.set_ylabel("Prawdziwa")
    ax.set_xlabel("Przewidziana")

plt.tight_layout()
plt.show()


# In[12]:


# Porównaj i zapisz lepszy model
rf_acc  = accuracy_score(y_test, y_pred_rf)
svm_acc = accuracy_score(y_test, y_pred_svm)

MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

if rf_acc >= svm_acc:
    best_name  = "Random Forest"
    joblib.dump(rf_model, os.path.join(MODELS_DIR, "posture_model.pkl"))
    joblib.dump(scaler,   os.path.join(MODELS_DIR, "scaler.pkl"))  # na wszelki wypadek
    print(f"✅ Zapisano Random Forest (accuracy: {rf_acc:.4f})")
else:
    best_name  = "SVM"
    joblib.dump(svm_model, os.path.join(MODELS_DIR, "posture_model.pkl"))
    joblib.dump(scaler,    os.path.join(MODELS_DIR, "scaler.pkl"))
    print(f"✅ Zapisano SVM (accuracy: {svm_acc:.4f})")

print(f"\nNajlepszy model: {best_name}")
print(f"Pliki zapisane w: {MODELS_DIR}")


# In[ ]:





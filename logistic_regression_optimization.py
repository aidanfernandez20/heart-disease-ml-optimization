# -*- coding: utf-8 -*-
"""
Logistic Regression Optimization Script
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# -------------------------------------------------------------------
# STEP 1: LOAD DATA
# -------------------------------------------------------------------
file_path = 'data/heart.csv'
print(f"Looking for file at: {os.path.abspath(file_path)}")

if not os.path.exists(file_path):
    print("Error: File not found.")
    raise FileNotFoundError(f"File not found at {file_path}")

df = pd.read_csv(file_path)
print("Data loaded successfully.")

# -------------------------------------------------------------------
# STEP 2: PREPROCESSING
# -------------------------------------------------------------------
TARGET_COLUMN = 'HeartDisease'

if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

y = df[TARGET_COLUMN]
X = df.drop(TARGET_COLUMN, axis=1)

# Handle Categorical Features
X = pd.get_dummies(X, drop_first=True)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data preprocessing complete.")

# -------------------------------------------------------------------
# STEP 3: HYPERPARAMETER TUNING
# -------------------------------------------------------------------
print("\n--- Starting Hyperparameter Tuning for Logistic Regression ---")

# Define the parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs'],
    'penalty': ['l2']
}

# Initialize the model
lr = LogisticRegression(random_state=42, max_iter=1000)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5, scoring='f1', verbose=1, n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train_scaled, y_train)

# -------------------------------------------------------------------
# STEP 4: EVALUATION
# -------------------------------------------------------------------
best_model = grid_search.best_estimator_
print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation F1 Score: {grid_search.best_score_:.4f}")

# Make predictions
y_pred = best_model.predict(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- Test Set Evaluation ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Optimized Logistic Regression Confusion Matrix")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

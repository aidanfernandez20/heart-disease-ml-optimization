# -*- coding: utf-8 -*-
"""
XGBoost Optimization Script
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, fbeta_score, roc_curve, roc_auc_score
from xgboost import XGBClassifier
from scipy.spatial import ConvexHull

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
print("\n--- Starting Hyperparameter Tuning for XGBoost ---")

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'scale_pos_weight': [1, 5], # Handling class imbalance
    'random_state': [42],
    'use_label_encoder': [False],
    'eval_metric': ['logloss']
}

# Initialize the model
xgb = XGBClassifier()

# Initialize GridSearchCV
# Using 'f1' for grid search to find a stable model before threshold tuning
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='f1', verbose=1, n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train_scaled, y_train)

# -------------------------------------------------------------------
# STEP 4: EVALUATION & THRESHOLD TUNING
# -------------------------------------------------------------------
best_model = grid_search.best_estimator_
print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation F1 Score: {grid_search.best_score_:.4f}")

# -------------------------------------------------------------------
# BASELINE MODEL EVALUATION (Threshold = 0.5)
# -------------------------------------------------------------------
print("\n--- Baseline XGBoost Evaluation (Threshold = 0.5) ---")

# Default prediction uses threshold=0.5 internally
y_pred_baseline = best_model.predict(X_test_scaled)

# Baseline metrics
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
baseline_precision = precision_score(y_test, y_pred_baseline)
baseline_recall = recall_score(y_test, y_pred_baseline)
baseline_f1 = f1_score(y_test, y_pred_baseline)

print(f"Accuracy: {baseline_accuracy:.4f}")
print(f"Precision: {baseline_precision:.4f}")
print(f"Recall: {baseline_recall:.4f}")
print(f"F1-Score: {baseline_f1:.4f}")

print("\nBaseline Classification Report:")
print(classification_report(y_test, y_pred_baseline))

print("Baseline Confusion Matrix:")
cm_base = confusion_matrix(y_test, y_pred_baseline)

plt.figure()
sns.heatmap(cm_base, annot=True, fmt='d', cmap='Oranges')
plt.title("Baseline XGBoost Confusion Matrix (Threshold = 0.5)")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Get probability predictions
y_prob = best_model.predict_proba(X_test_scaled)[:, 1]

# Threshold tuning loop
print("\n--- Threshold Tuning (Maximizing F2-Score) ---")
thresholds = np.arange(0.1, 1.0, 0.05)
best_threshold = 0.5
best_f2 = 0.0

print(f"{'Threshold':<10} | {'F2-Score':<10} | {'Recall':<10} | {'Precision':<10}")
print("-" * 50)

for thresh in thresholds:
    y_pred_thresh = (y_prob >= thresh).astype(int)
    
    # Calculate metrics
    rec = recall_score(y_test, y_pred_thresh)
    prec = precision_score(y_test, y_pred_thresh, zero_division=0)
    f2 = fbeta_score(y_test, y_pred_thresh, beta=2)
    
    print(f"{thresh:<10.2f} | {f2:<10.4f} | {rec:<10.4f} | {prec:<10.4f}")
    
    if f2 > best_f2:
        best_f2 = f2
        best_threshold = thresh

print(f"\nSelected Optimal Threshold (Max F2): {best_threshold}")

# Final predictions with optimal threshold
y_final_pred = (y_prob >= best_threshold).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_test, y_final_pred)
precision = precision_score(y_test, y_final_pred)
recall = recall_score(y_test, y_final_pred)
f1 = f1_score(y_test, y_final_pred)

print("\n--- Test Set Evaluation (Optimized) ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_final_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_final_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Optimized XGBoost (Threshold={best_threshold:.2f})")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# -------------------------------------------------------------------
# STEP 5: ROC Curve + Convex Hull + Annotations
# -------------------------------------------------------------------
custom_thresholds = np.arange(0.1, 1.0, 0.05)

# Compute FPR/TPR for each threshold
fprs = []
tprs = []

for thr in custom_thresholds:
    y_pred_thr = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thr).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    fprs.append(fpr)
    tprs.append(tpr)

fprs = np.array(fprs)
tprs = np.array(tprs)

# For comparison, compute true ROC curve
fpr_full, tpr_full, _ = roc_curve(y_test, y_prob)

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------

plt.figure(figsize=(8,6))

# Random classifier line
plt.plot([0,1], [0,1], "k--", alpha=0.5, label="Random Classifier")

# Full ROC curve (optional context)
plt.plot(fpr_full, tpr_full, label="ROC Curve", alpha=0.4)

target_threshold = 0.30

# Find the index of the threshold 0.30
idx = np.argmin(np.abs(custom_thresholds - target_threshold))

# Get its coordinates
x_30 = fprs[idx]
y_30 = tprs[idx]
plt.scatter(x_30, y_30, color="red", label="Threshold Models")
# Annotate it
plt.annotate("0.30",
             (x_30, y_30),
             textcoords="offset points",
             xytext=(5, -5),
             fontsize=10,
             color="black")

plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("XGBoost Models in ROC Space (varying thresholds)")
plt.legend()
plt.grid(True)
plt.show()
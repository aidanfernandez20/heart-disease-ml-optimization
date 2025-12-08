# -*- coding: utf-8 -*-
"""
PCA Analysis Script
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -------------------------------------------------------------------
# STEP 1: LOAD DATA
# -------------------------------------------------------------------
file_path = 'data/heart.csv'
print(f"Looking for file at: {os.path.abspath(file_path)}")

if not os.path.exists(file_path):
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
X_encoded = pd.get_dummies(X, drop_first=True)

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_encoded.columns)

print("Data preprocessing complete.")

# -------------------------------------------------------------------
# STEP 3: KURTOSIS CHECK
# -------------------------------------------------------------------
print("\n--- Kurtosis of Scaled Features ---")
# Fisher's definition (normal ==> 0.0) is used by scipy.stats.kurtosis by default
feature_kurtosis = X_scaled_df.apply(kurtosis)
print(feature_kurtosis.sort_values(ascending=False))

print("\n> Note: High positive kurtosis indicates heavy tails (outliers).")
print("> High negative kurtosis indicates light tails (flat).")
print("> Normal distribution has a kurtosis of 0 (Fisher's definition).")

# -------------------------------------------------------------------
# STEP 4: PCA ANALYSIS
# -------------------------------------------------------------------
print("\n--- Principal Component Analysis (PCA) ---")

# Fit PCA
pca = PCA()
pca.fit(X_scaled)

# Explained Variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("\nExplained Variance Ratio per Component:")
for i, var in enumerate(explained_variance):
    print(f"PC{i+1}: {var:.4f}")

# Check components needed for 90% and 95% variance
n_90 = np.argmax(cumulative_variance >= 0.90) + 1
n_95 = np.argmax(cumulative_variance >= 0.95) + 1

print(f"\nComponents needed for 90% variance: {n_90}")
print(f"Components needed for 95% variance: {n_95}")

# -------------------------------------------------------------------
# STEP 5: VISUALIZATION
# -------------------------------------------------------------------

# 1. Scree Plot
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.title('PCA Analysis')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# 2. 2D Projection (PC1 vs PC2)
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca_2d[:, 0], y=X_pca_2d[:, 1], hue=y, palette='viridis', alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: 2D Projection of Heart Disease Dataset')
plt.legend(title=TARGET_COLUMN)
plt.grid(True)
plt.show()

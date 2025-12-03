import pandas as pd
from scipy.stats import kurtosis
from sklearn.preprocessing import StandardScaler
import sys

# Load data
try:
    df = pd.read_csv('data/heart.csv')
except FileNotFoundError:
    print("Error: data/heart.csv not found")
    sys.exit(1)

# Preprocess
X = pd.get_dummies(df.drop('HeartDisease', axis=1), drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Calculate Kurtosis
k = X_scaled_df.apply(kurtosis).sort_values(ascending=False)
with open('kurtosis.txt', 'w') as f:
    f.write(k.to_string())
print("Kurtosis written to kurtosis.txt")

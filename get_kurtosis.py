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

# Plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
k.plot(kind='bar', color='skyblue', edgecolor='black')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Normal Distribution (0)')
plt.title('Feature Kurtosis values (Fisher)')
plt.ylabel('Kurtosis')
plt.xlabel('Features')
plt.legend()

# Add explanation text to the plot
explanation = (
    "High kurtosis (>3 or <-3) indicates heavy tails/outliers.\n"
    "PCA maximizes variance and is sensitive to outliers.\n"
    "Significant high kurtosis suggests data is not Gaussian,\n"
    "making PCA potentially unstable or misleading."
)
plt.gcf().text(0.15, 0.75, explanation, fontsize=10, color='darkred',
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))

plt.tight_layout()
plt.show()

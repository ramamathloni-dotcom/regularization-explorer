import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("data/telecom_churn.csv")

# Select numeric features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'churned' in numeric_cols:
    numeric_cols.remove('churned')

X = df[numeric_cols]
y = df['churned']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Generate 20 logarithmically spaced values for C
c_values = np.logspace(-3, 2, 20)

# Initialize lists to store coefficients
l1_coeffs = []
l2_coeffs = []

# Train models across the regularization spectrum
for c in c_values:
    # L1 Regularization
    model_l1 = LogisticRegression(penalty='l1', C=c, solver='saga', max_iter=5000)
    model_l1.fit(X_scaled, y)
    l1_coeffs.append(model_l1.coef_[0])
    
    # L2 Regularization
    model_l2 = LogisticRegression(penalty='l2', C=c, solver='saga', max_iter=5000)
    model_l2.fit(X_scaled, y)
    l2_coeffs.append(model_l2.coef_[0])

# Visualization
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# L1 Path Plot
ax[0].semilogx(c_values, l1_coeffs)
ax[0].set_title("L1 (Lasso) Regularization Path")
ax[0].set_xlabel("C (Log Scale)")
ax[0].set_ylabel("Coefficients")
ax[0].legend(numeric_cols, loc='best', fontsize='small')

# L2 Path Plot
ax[1].semilogx(c_values, l2_coeffs)
ax[1].set_title("L2 (Ridge) Regularization Path")
ax[1].set_xlabel("C (Log Scale)")
ax[1].set_ylabel("Coefficients")
ax[1].legend(numeric_cols, loc='best', fontsize='small')

plt.tight_layout()
plt.savefig("regularization_path.png")
plt.show()
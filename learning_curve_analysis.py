import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

data_path = os.path.join('data', 'telecom_churn.csv')
df = pd.read_csv(data_path)

target_col = 'churned'
y = df[target_col]

cols_to_drop = [target_col, 'customerID'] 
X = df.drop(columns=cols_to_drop, errors='ignore')

X = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X = scaler.fit_transform(X)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression(max_iter=1000)

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=cv, scoring='f1_macro', 
    train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.15)
plt.plot(train_sizes, val_mean, label='Cross-validation score', color='green', marker='o')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, color='green', alpha=0.15)

plt.title('Learning Curve (F1 Macro)')
plt.xlabel('Training Set Size')
plt.ylabel('F1 Macro Score')
plt.legend(loc='best')
plt.grid(True)

plt.savefig('learning_curve_result.png', dpi=300)
plt.show()
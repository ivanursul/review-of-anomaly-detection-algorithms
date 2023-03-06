import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('../resources/Mall_Customers.csv')
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit the One-Class SVM model
clf = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.05)
clf.fit(X_scaled)

# Predict outliers
y_pred = clf.predict(X_scaled)
outliers = X_scaled[y_pred == -1]

# Plot the data and outliers
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c='b', label='data points')
plt.scatter(outliers[:, 0], outliers[:, 1], c='r', label='outliers')
plt.title('One-Class SVM for Anomaly Detection')
plt.legend()
plt.show()
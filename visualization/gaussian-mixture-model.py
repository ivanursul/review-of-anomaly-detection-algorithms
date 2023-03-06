import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("../resources/Mall_Customers.csv")

# Select the relevant columns
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# Fit the Gaussian Mixture Clustering model to the data
model = GaussianMixture(n_components=5, covariance_type='full')
model.fit(X)

# Get the cluster labels
labels = model.predict(X)

# Identify the anomalies
anomalies = X[model.predict_proba(X).max(axis=1) < 0.5]

# Plot the results
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=labels, cmap='rainbow', alpha=0.7)
plt.scatter(anomalies.iloc[:,0], anomalies.iloc[:,1], color='black', marker='x', label='Anomalies')
plt.legend()
plt.show()
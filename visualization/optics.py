import pandas as pd
import numpy as np
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("../resources/Mall_Customers.csv")

# Select the relevant columns
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# Fit the OPTICS model to the data
model = OPTICS(min_samples=5, xi=.05, min_cluster_size=.05)
model.fit(X)

# Get the cluster labels
labels = model.labels_

# Identify the anomalies
anomalies = X[labels == -1]

# Plot the results
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=labels, cmap='rainbow', alpha=0.7)
plt.scatter(anomalies.iloc[:,0], anomalies.iloc[:,1], color='black', marker='x', label='Anomalies')
plt.legend()
plt.show()
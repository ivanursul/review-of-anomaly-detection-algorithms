import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("../resources/Mall_Customers.csv")

# Select the relevant columns
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# Estimate the bandwidth of the Mean Shift model
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

# Fit the Mean Shift model to the data
model = MeanShift(bandwidth=bandwidth)
model.fit(X)

# Get the cluster labels
labels = model.labels_

# Calculate the distance to the nearest cluster center for each point
distances = np.linalg.norm(X - model.cluster_centers_[model.labels_], axis=1)

# Set the anomaly threshold to the 95th percentile of the distances
threshold = np.percentile(distances, 95)

# Identify the anomalies
anomalies = X[distances > threshold]

# Plot the results
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=labels, cmap='rainbow', alpha=0.7)
plt.scatter(anomalies.iloc[:,0], anomalies.iloc[:,1], color='black', marker='x', label='Anomalies')
plt.legend()
plt.show()


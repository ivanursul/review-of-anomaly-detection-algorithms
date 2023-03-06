import pandas as pd
import numpy as np
import hdbscan

import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("Mall_Customers.csv")

# Select the relevant columns
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# Fit the HDBSCAN model to the data
model = hdbscan.HDBSCAN(min_cluster_size=5)
model.fit(X)

# Get the cluster labels
labels = model.labels_

# Calculate the distance to the nearest cluster center for each point
distances = model.outlier_scores_

# Set the anomaly threshold to the 95th percentile of the outlier scores
threshold = np.percentile(distances, 95)

# Identify the anomalies
anomalies = X[distances > threshold]

# Plot the results
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=labels, cmap='rainbow', alpha=0.7)
plt.scatter(anomalies.iloc[:,0], anomalies.iloc[:,1], color='black', marker='x', label='Anomalies')
plt.legend()
plt.show()

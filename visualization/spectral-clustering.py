import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("../resources/Mall_Customers.csv")

# Select the relevant columns
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# Fit the Spectral Clustering model to the data
model = SpectralClustering(n_clusters=5, affinity='nearest_neighbors', assign_labels='kmeans')
model.fit(X)

# Get the cluster labels
labels = model.labels_

# Identify the anomalies
anomalies = X[model.labels_ == -1]

# Plot the results
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=labels, cmap='rainbow', alpha=0.7)
plt.scatter(anomalies.iloc[:,0], anomalies.iloc[:,1], color='black', marker='x', label='Anomalies')
plt.legend()
plt.show()
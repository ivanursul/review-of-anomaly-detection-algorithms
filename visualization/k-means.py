import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time

# Load the data
data = pd.read_csv("../resources/Mall_Customers.csv")

# Select the relevant columns
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# Fit the K-Means model to the data
model = KMeans(n_clusters=5, init='k-means++')

start_time = time.time()
model.fit(X)
end_time = time.time()

# Calculate the total time taken for the KMeans algorithm to run
total_time = end_time - start_time

print("Total time taken: ", total_time)

# Get the cluster labels
labels = model.labels_

# Calculate the distance to the nearest cluster center for each point
distances = model.transform(X).min(axis=1)

# Set the anomaly threshold to the 95th percentile of the distances
threshold = np.percentile(distances, 95)

# Identify the anomalies
anomalies = X[distances > threshold]

# Plot the results
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=labels, cmap='rainbow', alpha=0.7)
plt.scatter(anomalies.iloc[:,0], anomalies.iloc[:,1], color='black', marker='x', label='Anomalies')
plt.legend()
plt.show()


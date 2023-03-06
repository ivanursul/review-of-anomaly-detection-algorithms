import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
import time
import numpy as np
import sys
#np.set_printoptions(threshold=sys.maxsize)

# Load creditcard.csv dataset
#data = pd.read_csv('../resources/creditcard.csv')
data = pd.read_csv('../resources/user_data.csv')


# Drop the 'Class' column since it's not needed for anomaly detection
X = data.drop(['Class'], axis=1)

# Set the parameters for the KMeans algorithm
n_clusters = 5

# Create a KMeans object with the specified parameters
kmeans = KMeans(n_clusters=n_clusters)

start_time = time.time()

# Fit the KMeans model and assign cluster labels to the data points
labels = kmeans.fit_predict(X)

# Calculate the distance to the nearest cluster center for each point
distances = kmeans.transform(X).min(axis=1)
#print("Distances: ", distances)

# Set the anomaly threshold to the 95th percentile of the distances
threshold = np.percentile(distances, 95)

print("Distance Threshold: ", threshold)

# Identify the anomalies
anomalies = X[distances > threshold]

# Calculate the precision, recall, and F1 metrics using scikit-learn's confusion_matrix function
true_labels = data['Class']

pred_labels = np.zeros(len(X))
pred_labels[anomalies.index] = 1

print("Number of Anomalies found: ", len(anomalies))
print("Number of real frauds: ", len(true_labels[true_labels == 1]))

tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

# Print the precision, recall, and F1 score
print("TN: ", tn, ", FP: ", fp, ", FN: ", fn, ", TP:", tp)
print("Duration: %s seconds" % (time.time() - start_time))
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1_score)

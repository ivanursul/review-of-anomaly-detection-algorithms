import pandas as pd
import hdbscan
import numpy as np
from sklearn.metrics import confusion_matrix
import time

# Load creditcard.csv dataset
data = pd.read_csv('../resources/creditcard.csv')

# Drop the 'Class' column since it's not needed for anomaly detection
X = data.drop(['Class'], axis=1)

# Set the parameters for the HDBSCAN algorithm
min_cluster_size = 5
min_samples = 1

start_time = time.time()

# Create an HDBSCAN object with the specified parameters
hdbscan = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)

# Fit the DBSCAN model and assign cluster labels to the data points
hdbscan.fit(X)

# Get the predicted cluster labels
labels = hdbscan.labels_

# Calculate the distance to the nearest cluster center for each point
distances = hdbscan.outlier_scores_

# Set the anomaly threshold to the 95th percentile of the outlier scores
threshold = np.percentile(distances, 95)

# Identify the anomalies
anomalies = X[distances > threshold]


# Calculate the precision, recall, and F1 metrics using scikit-learn's confusion_matrix function
true_labels = data['Class']

pred_labels = labels.copy()
pred_labels[pred_labels != -1] = 0
pred_labels[pred_labels == -1] = 1

#print("Pred labels %s " % true_labels)

tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()

print("TN: ", tn, ", FP: ", fp, ", FN: ", fn, ", TP:", tp)

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

# Print the precision, recall, and F1 score
print("Number of Anomalies found: ", len(anomalies))
print("Number of real frauds: ", len(true_labels[true_labels == 1]))
print("Duration: %s seconds" % (time.time() - start_time))
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1_score)
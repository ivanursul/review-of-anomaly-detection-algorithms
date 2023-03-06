import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import time

# Load metrics.csv dataset
data = pd.read_csv('../resources/creditcard.csv')

# Drop the 'Class' column since it's not needed for anomaly detection
X = data.drop(['Class'], axis=1)

# Set the parameters for the MiniBatchKMeans algorithm
n_clusters = 5

start_time = time.time()

# Create a MiniBatchKMeans object with the specified parameters
mbkmeans = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', batch_size=1000)

# Fit the MiniBatchKMeans model and assign cluster labels to the data points
labels = mbkmeans.fit_predict(X)

# Calculate the distance to the nearest cluster center for each point
distances = mbkmeans.transform(X).min(axis=1)

# Set the anomaly threshold to the 95th percentile of the distances
threshold = np.percentile(distances, 95)

# Identify the anomalies
anomalies = X[distances > threshold]
print("Number of Anomalies found: ", len(anomalies))

# Calculate the precision, recall, and F1 metrics using scikit-learn's confusion_matrix function
true_labels = data['Class']
print("Number of real frauds: ", len(true_labels[true_labels == 1]))

pred_labels = labels.copy()
pred_labels[pred_labels != -1] = 0
pred_labels[pred_labels == -1] = 1
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

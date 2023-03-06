import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import confusion_matrix
import time
import numpy as np

# Load creditcard.csv dataset
data = pd.read_csv('../resources/hypothyroid.csv')

# Drop the 'Class' column since it's not needed for anomaly detection
X = data.drop('class', axis=1)

from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()
for x in X.columns:
    X[x]=enc.fit_transform(X[x])
X.info()

# Set the bandwidth parameter for the MeanShift algorithm
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

start_time = time.time()

# Create a MeanShift object with the specified bandwidth parameter
meanshift_model = MeanShift(bandwidth=bandwidth)

# Fit the MeanShift model and assign cluster labels to the data points
labels = meanshift_model.fit_predict(X)

# Calculate the distance to the nearest cluster center for each point
distances = np.linalg.norm(X - meanshift_model.cluster_centers_[meanshift_model.labels_], axis=1)

# Set the anomaly threshold to the 95th percentile of the distances
threshold = np.percentile(distances, 95)

# Identify the anomalies
anomalies = X[distances > threshold]

print("Anomalies: ", anomalies)

# Compare the anomalies with the 'fraud' instances in the 'Class' column using a confusion matrix
true_labels = data['class'].replace({'P': 1, 'N': 0})
pred_labels = [1 if index in anomalies.index else 0 for index in data.index]
tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
print("TN: ", tn, ", FP: ", fp, ", FN: ", fn, ", TP:", tp)

# Print the confusion matrix
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

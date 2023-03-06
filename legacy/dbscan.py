import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix, classification_report

# Load metrics.csv dataset
data = pd.read_csv('../resources/user_data.csv')

# Drop the 'Class' column since it's not needed for anomaly detection
X = data.drop(['Class'], axis=1)

# Set the parameters for the DBSCAN algorithm
eps = 0.5
min_samples = 5

# Create a DBSCAN object with the specified parameters
dbscan = DBSCAN(eps=eps, min_samples=min_samples)

# Fit the DBSCAN model and assign cluster labels to the data points
dbscan.fit(X)

# Get the predicted cluster labels
labels = dbscan.labels_

# Identify the anomalous data points with a label of -1
anomalies = X[labels == -1]

# Calculate the precision, recall, and F1 metrics using scikit-learn's confusion_matrix function
true_labels = data['Class']
pred_labels = labels.copy()
pred_labels[pred_labels != -1] = 0
pred_labels[pred_labels == -1] = 1
tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

# Print the precision, recall, and F1 score
print("Number of Anomalies found: ", len(anomalies))
print("Number of real frauds: ", len(true_labels[true_labels == 1]))
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1_score)
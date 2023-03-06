import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix, classification_report
import time

# Load metrics.csv dataset
data = pd.read_csv('../resources/creditcard.csv')

# Drop the 'Class' column since it's not needed for anomaly detection
X = data.drop(['Class'], axis=1)

# Create an AgglomerativeClustering object with the specified parameters
agc = AgglomerativeClustering(n_clusters=5, linkage='single')

start_time = time.time()

# Fit the AgglomerativeClustering model and assign cluster labels to the data points
agc.fit(X)

# Get the predicted cluster labels
labels = agc.labels_

# Identify the anomalous data points with a label of -1
anomalies = X[labels == -1]

# Calculate the precision, recall, and F1 metrics using scikit-learn's confusion_matrix function
true_labels = data['Class']
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
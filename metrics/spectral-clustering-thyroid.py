import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import confusion_matrix, classification_report
import time

# Load metrics.csv dataset
data = pd.read_csv('../resources/hypothyroid.csv')

# Drop the 'Class' column since it's not needed for anomaly detection
X = data.drop('class', axis=1)

from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()
for x in X.columns:
    X[x]=enc.fit_transform(X[x])
X.info()

# Set the parameters for the SpectralClustering algorithm
start_time = time.time()
n_clusters = 8
affinity = 'nearest_neighbors'

# Create a SpectralClustering object with the specified parameters
spectral = SpectralClustering(n_clusters=n_clusters, affinity=affinity)

# Fit the SpectralClustering model and assign cluster labels to the data points
labels = spectral.fit_predict(X)

# Identify the anomalous data points with a label of -1
anomalies = X[labels == -1]

# Calculate the precision, recall, and F1 metrics using scikit-learn's confusion_matrix function
true_labels = data['class'].replace({'P': 1, 'N': 0})
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
print("TN: ", tn, ", FP: ", fp, ", FN: ", fn, ", TP:", tp)
print("Duration: %s seconds" % (time.time() - start_time))
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1_score)

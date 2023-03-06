import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix, classification_report
import time

# Load metrics.csv dataset
data = pd.read_csv('../resources/creditcard.csv')

# Drop the 'Class' column since it's not needed for anomaly detection
X = data.drop(['Class'], axis=1)

# Set the parameters for the LocalOutlierFactor algorithm
n_neighbors = 20
contamination = 0.1

start_time = time.time()

# Create a LocalOutlierFactor object with the specified parameters
lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)

# Fit the LocalOutlierFactor model and get the predicted outlier scores
scores = lof.fit_predict(X)

# Identify the anomalous data points with a score less than -1
anomalies = X[scores < -1]

# Calculate the precision, recall, and F1 metrics using scikit-learn's confusion_matrix function
true_labels = data['Class']
pred_labels = scores.copy()
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
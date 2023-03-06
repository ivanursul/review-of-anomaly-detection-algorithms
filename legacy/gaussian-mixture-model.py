import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, classification_report
import time
# Load metrics.csv dataset
data = pd.read_csv('../resources/creditcard.csv')

# Drop the 'Class' column since it's not needed for anomaly detection
X = data.drop(['Class'], axis=1)

# Set the parameters for the GaussianMixture algorithm
start_time = time.time()

# Create a GaussianMixture object with the specified parameters
gm = GaussianMixture(n_components=5, covariance_type='full')

# Fit the GaussianMixture model and assign cluster labels to the data points
gm.fit(X)
labels = gm.predict(X)

# Identify the anomalies
anomalies = X[gm.predict_proba(X).max(axis=1) < 0.5]

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
print("TN: ", tn, ", FP: ", fp, ", FN: ", fn, ", TP:", tp)
print("Duration: %s seconds" % (time.time() - start_time))
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1_score)
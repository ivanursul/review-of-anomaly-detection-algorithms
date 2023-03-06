import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, classification_report

# Load metrics.csv dataset
data = pd.read_csv('../resources/hypothyroid.csv')

# Drop the 'Class' column since it's not needed for anomaly detection
X = data.drop('class', axis=1)

from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()
for x in X.columns:
    X[x]=enc.fit_transform(X[x])
X.info()


# Set the parameters for the OneClassSVM algorithm
nu = 0.1
kernel = 'rbf'

# Create a OneClassSVM object with the specified parameters
ocsvm = OneClassSVM(nu=nu, kernel=kernel)

# Fit the OneClassSVM model and get the predicted outlier scores
scores = ocsvm.fit_predict(X)

# Identify the anomalous data points with a score less than 0
anomalies = X[scores < 0]

# Calculate the precision, recall, and F1 metrics using scikit-learn's confusion_matrix function
true_labels = data['class'].replace({'P': 1, 'N': 0})
pred_labels = scores.copy()
pred_labels[pred_labels >= 0] = 0
pred_labels[pred_labels < 0] = 1
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

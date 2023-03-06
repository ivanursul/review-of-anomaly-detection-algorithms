import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import time

# Load the data into a pandas dataframe
data = pd.read_csv('../resources/hypothyroid.csv')

df = data.drop('class', axis=1)

from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()
for x in df.columns:
    df[x]=enc.fit_transform(df[x])
df.info()

# Split the data into features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

start_time = time.time()

# Perform PCA to reduce the dimensionality of the data
# pca = PCA(n_components=2, random_state=42)
# X_pca = pca.fit_transform(X)

# Scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform k-means clustering with k=2
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)

# Get the distances of each point to the nearest centroid
distances = kmeans.transform(X_scaled).min(axis=1)

threshold = np.percentile(distances, 95)

print("Distance Threshold: ", threshold)

# Identify the anomalies
anomalies = X[distances > threshold]

print(anomalies)

# Calculate the precision, recall, and F1 metrics using scikit-learn's confusion_matrix function
true_labels = data['class'].replace({'P': 1, 'N': 0})
print("True labels length: ", len(true_labels[true_labels == 1]), ", Total Items: ", len(data['class']))

pred_labels = np.zeros(len(X))
pred_labels[anomalies.index] = 1

print("Number of Anomalies found: ", len(anomalies))
print("Number of real cases: ", len(true_labels[true_labels == 1]))

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

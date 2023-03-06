import pandas as pd
from sklearn.cluster import DBSCAN
import pyarrow
import fastparquet
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load metrics.csv dataset
data = pd.read_parquet('../resources/running/run_ww_2019_d.parquet')

# Drop the 'Class' column since it's not needed for anomaly detection
data = data.drop(['datetime', 'major', 'athlete', 'gender', 'age_group', 'country'], axis=1)

print("DataSet size: ", len(data))

data = data[(data['distance'] > 0) & (data['duration'] > 0)]


sample = data.sample(n=100000)

fake_running_activity = {'distance': 100.00, 'duration': 5000}
sample = sample.append(fake_running_activity, ignore_index=True)

#sample = data.copy()
#sample = sample[["distance", "duration", "gender"]]
print("Filtered Data Set Size: ", len(sample))


X = sample.copy()

# enc = LabelEncoder()
# for x in ['gender', 'age_group', 'country']:
#     X[x]=enc.fit_transform(X[x])
# X.info()

# Set the parameters for the DBSCAN algorithm
eps = 0.5
min_samples = 5

# Create a DBSCAN object with the specified parameters
dbscan = DBSCAN(eps=eps, min_samples=min_samples)

# Fit the DBSCAN model and assign cluster labels to the data points
dbscan.fit(X)

# Get the predicted cluster labels
labels = dbscan.labels_
print("Labels: ", labels)

# Identify the anomalous data points with a label of -1
anomalies = X[labels == -1]

# Print the precision, recall, and F1 score
print("Number of Anomalies found: ", len(anomalies))
print("Anomalies: \n", anomalies)
print("Fake anomaly detection result: ", labels[len(sample) - 1])

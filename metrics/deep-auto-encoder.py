import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.layers import Input, Dense
from keras.models import Model
import time

# Load metrics.csv dataset
data = pd.read_csv('../resources/hypothyroid.csv')

# Drop the 'Class' column since it's the target variable
X_raw = data.drop(['class'], axis=1)
y = data['class'].replace({'P': 1, 'N': 0})


from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()
for x in X_raw.columns:
    X_raw[x]=enc.fit_transform(X_raw[x])
X_raw.info()

X = X_raw.copy()

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data to have zero mean and unit variance
X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)
X_train_norm = (X_train - X_train_mean) / X_train_std

# Define the deep autoencoder architecture
input_dim = X_train_norm.shape[1]
encoding_dim = 10
input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)
autoencoder = Model(input_layer, decoded)
start_time = time.time()

# Train the autoencoder on the train set
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(X_train_norm, X_train_norm, epochs=50, batch_size=32)

# Use the trained autoencoder to predict the reconstruction error for the test set
X_test_norm = (X_test - X_train_mean) / X_train_std
y_pred_norm = autoencoder.predict(X_test_norm)
mse = np.mean(np.power(X_test_norm - y_pred_norm, 2), axis=1)
y_pred = np.zeros_like(y_test)
y_pred[mse > np.percentile(mse, 95)] = 1

# Calculate the precision, recall, and F1 metrics using scikit-learn's confusion_matrix function
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

# Print the precision, recall, and F1 score
print("Number of Anomalies found: ", len(y_pred))
print("Number of real frauds: ", len(y_test[y_test == 1]))

print("TN: ", tn, ", FP: ", fp, ", FN: ", fn, ", TP:", tp)
print("Duration: %s seconds" % (time.time() - start_time))
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1_score)

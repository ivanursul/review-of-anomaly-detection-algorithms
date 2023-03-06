import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load data
df = pd.read_csv("../resources/Mall_Customers.csv")
X = df.iloc[:, 3:].values  # We only need the 'Annual Income' and 'Spending Score' columns
y = df.iloc[:, -1].values

# Normalize data
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Define auto-encoder model
input_dim = X.shape[1]
hidden_dim = 2  # 2-dimensional latent space for easy visualization
output_dim = X.shape[1]

input_layer = tf.keras.layers.Input(shape=(input_dim,))
hidden_layer = tf.keras.layers.Dense(hidden_dim, activation='relu')(input_layer)
output_layer = tf.keras.layers.Dense(output_dim, activation='linear')(hidden_layer)

autoencoder = tf.keras.Model(inputs=input_layer, outputs=output_layer)

# Compile model
autoencoder.compile(optimizer='adam', loss='mse')

# Train model
history = autoencoder.fit(X, X, epochs=50, batch_size=32, validation_split=0.2)

# Make predictions and compute reconstruction error
X_pred = autoencoder.predict(X)
mse = np.mean(np.power(X - X_pred, 2), axis=1)

# Compute anomaly score
threshold = np.percentile(mse, 95)
anomaly_score = np.where(mse > threshold, 1, 0)

# Visualize data and anomalies
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.scatter(X[anomaly_score == 1, 0], X[anomaly_score == 1, 1], c='r', marker='x', s=50)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()
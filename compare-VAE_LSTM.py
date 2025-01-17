import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

# Define the Variational Autoencoder (VAE)
class VAE(Model):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(input_dim,)),
            layers.Dense(64, activation='tanh'),
            layers.Dense(32, activation='tanh'),
            layers.Dense(latent_dim + latent_dim),  # Mean and log variance
        ])
        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(32, activation='tanh'),
            layers.Dense(64, activation='tanh'),
            layers.Dense(input_dim, activation='sigmoid'),
        ])

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z):
        return self.decoder(z)

    def call(self, inputs):
        mean, logvar = self.encode(inputs)
        z = self.reparameterize(mean, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mean, logvar

# Define the LSTM model for sequence prediction
class LSTMModel(Model):
    def __init__(self, latent_dim):
        super(LSTMModel, self).__init__()
        self.lstm = layers.LSTM(64, return_sequences=True, input_shape=(None, latent_dim))
        self.dense = layers.Dense(latent_dim)

    def call(self, inputs):
        x = self.lstm(inputs)
        return self.dense(x)

# Loss function for VAE
def vae_loss(x, reconstruction, mean, logvar):
    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - reconstruction), axis=1))
    kl_divergence = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar))
    return reconstruction_loss + kl_divergence

# Training the VAE-LSTM hybrid model
def train_vae_lstm(vae, lstm, data, batch_size=32, epochs=100):
    optimizer = tf.keras.optimizers.Adam()
    sequence_length = 3  # Ensure consistent sequence_length
    for epoch in range(epochs):
        for i in range(0, len(data) - batch_size, batch_size):
            batch = data[i:i + batch_size]
            batch = tf.convert_to_tensor(batch, dtype=tf.float32)
            with tf.GradientTape() as tape:
                reconstruction, mean, logvar = vae(batch)
                embeddings = vae.reparameterize(mean, logvar)

                # Adjust batch_size for reshape
                num_samples = embeddings.shape[0]
                trim_size = (num_samples // sequence_length) * sequence_length  # Ensure divisibility
                embeddings = embeddings[:trim_size]  # Trim extra samples
                embeddings = tf.reshape(embeddings, (-1, sequence_length, embeddings.shape[-1]))  # Reshape for LSTM

                lstm_output = lstm(embeddings)
                lstm_loss = tf.reduce_mean(tf.square(lstm_output - embeddings))
                loss = vae_loss(batch, reconstruction, mean, logvar) + lstm_loss

            gradients = tape.gradient(loss, vae.trainable_variables + lstm.trainable_variables)
            optimizer.apply_gradients(zip(gradients, vae.trainable_variables + lstm.trainable_variables))
        print(f"Epoch {epoch+1}, Loss: {loss.numpy()}")

# Example: Calculate Silhouette Score
def calculate_silhouette_score(vae, data, n_clusters=3):
    # Extract embeddings from the VAE encoder
    mean, logvar = vae.encode(data)
    embeddings = vae.reparameterize(mean, logvar).numpy()

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    labels = kmeans.labels_

    # Calculate Silhouette Score
    score = silhouette_score(embeddings, labels)
    print(f"Silhouette Score: {score}")
    return score

# Example: Calculate MSE and MAE
def calculate_mse_mae(vae, lstm, data):
    # VAE reconstruction error
    reconstruction, _, _ = vae(data)
    mse = np.mean(np.square(data - reconstruction.numpy()), axis=1)
    mae = np.mean(np.abs(data - reconstruction.numpy()), axis=1)
    print(f"VAE MSE: {np.mean(mse)}, VAE MAE: {np.mean(mae)}")

    # LSTM prediction error
    mean, logvar = vae.encode(data)
    embeddings = vae.reparameterize(mean, logvar)
    lstm_predictions = lstm(tf.reshape(embeddings, (-1, data.shape[0] // 4, embeddings.shape[-1])))

    # Reshape LSTM predictions back to 2D for comparison
    reshaped_lstm_predictions = tf.reshape(lstm_predictions, (-1, embeddings.shape[-1]))
    # Calculate MSE between reshaped predictions and original embeddings
    lstm_mse = np.mean(np.square(reshaped_lstm_predictions.numpy() - embeddings.numpy()), axis=1)
    lstm_mae = np.mean(np.abs(reshaped_lstm_predictions.numpy() - embeddings.numpy()), axis=1)

    # lstm_mse = np.mean(np.square(lstm_predictions.numpy() - embeddings.numpy()), axis=1)
    # lstm_mae = np.mean(np.abs(lstm_predictions.numpy() - embeddings.numpy()), axis=1)
    print(f"LSTM MSE: {np.mean(lstm_mse)}, LSTM MAE: {np.mean(lstm_mae)}")

    return mse, mae, lstm_mse, lstm_mae

# Example: Define anomaly detection threshold
def calculate_threshold(errors, percentile=95):
    # Use a given percentile of the normal error distribution as the threshold
    threshold = np.percentile(errors, percentile)
    print(f"Anomaly Detection Threshold (at {percentile}th percentile): {threshold}")
    return threshold

if __name__ == "__main__":
    # Example synthetic dataset
    input_dim = 9  # Number of features in each time step
    latent_dim = 2  # Dimensionality of the latent space
    # time_series_length = 1000

    data = pd.read_csv('./data/smart_info_data_augmented.csv')
    data.set_index('DateTime', inplace=True)


    # Scale the data (normalization)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Create sequences for training
    def create_sequences(data, sequence_length):
        sequences = []
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:i + sequence_length])
        return np.array(sequences)

    sequence_length = 3  # Adjust this value based on your needs
    sequences = create_sequences(scaled_data, sequence_length)

    # Split the sequences into train and test sets
    train_size = int(len(sequences) * 0.8)
    train_sequences = sequences[:train_size]
    test_sequences = sequences[train_size:]

    # Define input dimensions based on processed data
    input_dim = train_sequences.shape[-1]  # Number of features
    latent_dim = 2  # Dimensionality of latent space

    # Initialize and train models
    vae = VAE(input_dim, latent_dim)
    lstm = LSTMModel(latent_dim)
    train_vae_lstm(vae, lstm, scaled_data, batch_size=32, epochs=100)

    # Compute Silhouette Score
    silhouette_score_value = calculate_silhouette_score(vae, scaled_data, n_clusters=2)

    # Compute MSE and MAE
    vae_mse, vae_mae, lstm_mse, lstm_mae = calculate_mse_mae(vae, lstm, scaled_data)

    # Compute anomaly detection threshold
    anomaly_threshold = calculate_threshold(vae_mse + lstm_mse, percentile=95)

    print(f"silhouette_score_value : {silhouette_score_value:.6f}")
    # print(f"vae_mae,  lstm_mae, vae_mse, lstm_mse : {vae_mae:.6f}, {lstm_mae:.6f}, {vae_mse:.6f}, {lstm_mse:.6f}")
    print(f"anomaly_threshold : {anomaly_threshold:.6f}")
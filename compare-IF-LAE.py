from datetime import datetime
import numpy as np
import pandas as pd

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, mean_squared_error, mean_absolute_error


# 시계열 데이터 윈도우 생성
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)


# Load the unlabeled dataset
if __name__ == "__main__":

    input_dim = 9
    sequence_length = 3
    epochs = 100
    batch_size = 32

    file_time = datetime.now().strftime("%Y%m%d%H%M%S")

    data = pd.read_csv('./data/smart_info_data_augmented.csv')
    data.set_index('DateTime', inplace=True)

    # Preprocess the data
    numerical_features = [
        'Temperature', 'Percentage_Used', 'Host_Read_Commands', 'Host_Write_Commands',
        'Controller_Busy_Time', 'Data_Units_Read_Numbers', 'Data_Units_Read_Bytes',
        'Data_Units_Written_Numbers', 'Data_Units_Written_Bytes'
    ]

    scaled_data = MinMaxScaler().fit_transform(data)
    # data_scaled = data_scaled.reshape(data_scaled.shape[0], 1, data_scaled.shape[1])  # Reshape for LSTM

    # 1. Train Isolation Forest
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    isolation_scores = iso_forest.fit_predict(scaled_data)
    normal_data = scaled_data[isolation_scores == 1]
    anomaly_data = scaled_data[isolation_scores == -1]

    scores = iso_forest.decision_function(normal_data)
    threshold = np.percentile(scores, 5)

    filtered_data = create_sequences(normal_data[scores >= threshold], sequence_length)  # 점수가 Threshold 이상인 데이터만 사용


    print(f"scores / threshold : {scores}, {threshold}")

    # Calculate Silhouette Score for Isolation Forest
    silhouette = silhouette_score(data[numerical_features], isolation_scores)


    # 2. Define LSTM Autoencoder Model
    def create_lstm_autoencoder(seq_len, num_features):
        model = Sequential([
            LSTM(32, activation='relu', input_shape=(seq_len, num_features), return_sequences=True),
            LSTM(16, activation='relu', return_sequences=False),
            RepeatVector(seq_len),
            LSTM(16, activation='relu', return_sequences=True),
            LSTM(32, activation='relu', return_sequences=True),
            TimeDistributed(Dense(num_features))
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    # Create and train the LSTM Autoencoder
    # seq_len, num_features = data_scaled.shape[1], data_scaled.shape[2]
    lstm_autoencoder = create_lstm_autoencoder(sequence_length, input_dim)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

    model_file_name = './model/best_model-IF-LAE-{}.h5'.format(file_time)
    mc = ModelCheckpoint(model_file_name, monitor='val_loss', mode='min', save_best_only=True)

    lstm_autoencoder.fit(
        filtered_data, filtered_data,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
        callbacks=[es, mc]
    )

    # 3. Anomaly Detection Using Reconstruction Error
    reconstructions = lstm_autoencoder.predict(filtered_data)

    mse = np.mean(np.power(filtered_data - reconstructions, 2), axis=(1, 2))
    mae = np.mean(np.abs(filtered_data - reconstructions), axis=(1, 2))



    threshold = np.percentile(mse, 95)  # Use the 95th percentile as the anomaly threshold

    lstm_anomalies = mse > threshold

    print(f"Silhouette Score for Isolation Forest: {silhouette:.6f}")
    print(f"MSE Threshold for LSTM Autoencoder: {threshold:.6f}")

    # Calculate MSE and MAE
    print(f"Mean Squared Error (MSE): {np.mean(mse):.6f}")
    print(f"Mean Absolute Error (MAE): {np.mean(mae):.6f}")
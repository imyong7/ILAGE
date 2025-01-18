from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, Reshape, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans, MiniBatchKMeans

# Function to calculate Silhouette Score
def calculate_silhouette(data, labels):
    if len(np.unique(labels)) > 1:  # Ensure there is more than one cluster
        return silhouette_score(data, labels)
    else:
        return None


# noinspection PyPackageRequirements
if __name__ == "__main__":
    print("GPU available:", tf.test.is_gpu_available())
    input_dim = 9
    sequence_length = 3
    epochs = 100
    batch_size = 32

    file_time = datetime.now().strftime("%Y%m%d%H%M%S")
    start_time = datetime.now()
    df_datetime = {}
    df = pd.read_csv('./data/smart_info_data_augmented.csv')
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df_datetime = df['DateTime'].copy(deep=True)
    df.set_index("DateTime", inplace=True)

    numerical_features = [
        'Temperature', 'Percentage_Used', 'Host_Read_Commands', 'Host_Write_Commands',
        'Controller_Busy_Time', 'Data_Units_Read_Numbers', 'Data_Units_Read_Bytes',
        'Data_Units_Written_Numbers', 'Data_Units_Written_Bytes'
    ]

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)


    # 시계열 데이터 윈도우 생성
    def create_sequences(data, sequence_length):
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        return np.array(sequences)

    # 시퀀스 데이터 생성
    sequences = create_sequences(scaled_data, sequence_length)
    print(f"Sequences shape: {sequences.shape}")  # (샘플 수, 시퀀스 길이, 피처 수)


    # Isolation Forest
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    outlier_labels = iso_forest.fit_predict(scaled_data)  # scaled_data는 정규화된 데이터
    normal_data = scaled_data[outlier_labels == 1]
    anomaly_data = scaled_data[outlier_labels == -1]

    scores = iso_forest.decision_function(normal_data)
    threshold = np.percentile(scores, 5)

    print(f"scores / threshold : {scores}, {threshold}")


    # 정상 데이터 필터링
    # filtered_data = normal_data[scores >= threshold]  # 점수가 Threshold 이상인 데이터만 사용
    filtered_data = create_sequences(normal_data[scores >= threshold], sequence_length)  # 점수가 Threshold 이상인 데이터만 사용

    # Isolation Forest 모델을 사용해 filtered_data 예측
    # flatten_filtered_data = np.reshape(filtered_data.shape[0], filtered_data.shape[2])
    flatten_filtered_data = filtered_data[:, 0, :]

    # print(flatten_filtered_data.shape)

    outlier_labels_filtered = iso_forest.predict(flatten_filtered_data)



    # LSTM-AutoEncoder
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

    model_file_name = './model/best_model-ILAGE-{}.h5'.format(file_time)
    mc = ModelCheckpoint(model_file_name, monitor='val_loss', mode='min', save_best_only=True)

    input_layer = Input(shape=(sequence_length, input_dim))
    encoded = LSTM(64, activation='relu', return_sequences=True)(input_layer)
    encoded = LSTM(32, activation='relu', return_sequences=False)(encoded)

    repeated = RepeatVector(sequence_length)(encoded)

    decoded = LSTM(32, activation='relu', return_sequences=True)(repeated)
    decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.summary()

    # 모델 학습
    history = autoencoder.fit(
        filtered_data, filtered_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[es, mc]
    )

    # LSTM-AutoEncoder 복원 오차 계산
    # lstm_scores = autoencoder.predict(filtered_data)

    # 복원 데이터 계산
    reconstructed_data = autoencoder.predict(filtered_data)

    # 복원 오차 계산
    # reconstruction_errors = np.mean(np.abs(filtered_data - reconstructed_data), axis=1)


    # Generator 정의
    generator = Sequential([
        # Dense(64, activation='relu', input_dim=input_dim),
        # Dense(filtered_data.shape[1], activation='sigmoid')
        Dense(32, activation='relu', input_dim=input_dim),
        Dense(sequence_length * input_dim, activation='sigmoid'),  # 3 (시퀀스 길이) * 9 (피처 수)
        Reshape((sequence_length, input_dim))  # 출력 차원을 (시퀀스 길이, 피처 수)로 변환
    ])

    # Discriminator 정의
    discriminator = Sequential([
        Flatten(input_shape=(sequence_length, input_dim)),  # 3차원을 2차원으로 평탄화
        Dense(32, activation='relu', input_dim=filtered_data.shape[1]),
        Dense(1, activation='sigmoid')
    ])

    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # GAN 모델 정의
    gan = Sequential([generator, discriminator])
    gan.compile(optimizer='adam', loss='binary_crossentropy')

    # GAN 학습 루프
    for epoch in range(epochs):
        # Generate fake data
        noise = np.random.normal(0, 1, (batch_size, input_dim))
        generated_data = generator.predict(noise)

        # Concatenate real and fake data
        combined_data = np.vstack((filtered_data, generated_data))
        labels = np.hstack((np.ones(filtered_data.shape[0]), np.zeros(generated_data.shape[0])))

        # Train Discriminator
        discriminator.trainable = True
        discriminator.train_on_batch(combined_data, labels)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, input_dim))
        discriminator.trainable = False
        gan.train_on_batch(noise, np.ones(batch_size))

    # LSTM-AutoEncoder와 GAN의 결과 병합
    lstm_scores = autoencoder.predict(filtered_data)
    gan_scores = discriminator.predict(filtered_data)

    gan_scores_expanded = gan_scores[:, np.newaxis, np.newaxis]  # 차원 확장
    gan_scores_expanded = np.repeat(gan_scores_expanded, lstm_scores.shape[1], axis=1)
    gan_scores_expanded = np.repeat(gan_scores_expanded, lstm_scores.shape[2], axis=2)
    gan_scores_expanded = gan_scores_expanded.squeeze()

    # Isolation Forest 점수 (-1: 이상치, 1: 정상치)
    iso_forest_scores = (outlier_labels_filtered == 1).astype(int)  # 정상 데이터: 1, 이상치: 0
    iso_forest_scores_expanded = iso_forest_scores[:, np.newaxis, np.newaxis]  # 차원 확장
    iso_forest_scores_expanded = np.repeat(iso_forest_scores_expanded, lstm_scores.shape[1], axis=1)
    iso_forest_scores_expanded = np.repeat(iso_forest_scores_expanded, lstm_scores.shape[2], axis=2)

    ensemble_scores = 0.5 * iso_forest_scores_expanded + 0.2 * lstm_scores + 0.1 * gan_scores_expanded

    # 임계값 설정 (상위 95% 이상치)
    ensemble_threshold = np.percentile(ensemble_scores, 95)

    # 이상치 탐지
    anomalies = ensemble_scores > ensemble_threshold

    # 결과를 DataFrame에 저장
    # 1. 3차원 데이터를 2차원으로 축소
    iso_forest_scores_flattened = np.mean(iso_forest_scores_expanded, axis=1)  # 시퀀스 길이에 대해 평균
    lstm_scores_flattened = np.mean(lstm_scores, axis=1)
    gan_scores_flattened = np.mean(gan_scores_expanded, axis=1)
    ensemble_scores_flattened = np.mean(ensemble_scores, axis=1)
    anomalies_flattened = np.any(anomalies, axis=(1, 2))  # 샘플별 이상 탐지 여부 (True/False)

    # 2. DataFrame 생성
    results_df = pd.DataFrame({
        "Isolation Forest Score": iso_forest_scores_flattened.mean(axis=1),  # 피처별 평균
        "LSTM-AutoEncoder Score": lstm_scores_flattened.mean(axis=1),
        "GAN Score": gan_scores_flattened.mean(axis=1),
        "Ensemble Score": ensemble_scores_flattened.mean(axis=1),
        "Anomaly": anomalies_flattened.astype(int)  # 이상 탐지 여부 (1/0)
    })

    # 3. 결과 확인
    # print(results_df)

    # 결과 출력
    plt.figure(figsize=(12, 6))

    # `results_df`에서 점수를 가져와 시각화
    plt.plot(results_df.index, results_df["Isolation Forest Score"], label="Isolation Forest", linestyle='-', color='green')
    plt.plot(results_df.index, results_df["LSTM-AutoEncoder Score"], label="LSTM-AutoEncoder", linestyle='--', color='blue')
    plt.plot(results_df.index, results_df["GAN Score"], label="GAN", linestyle='-.', color='orange')
    plt.plot(results_df.index, results_df["Ensemble Score"], label="Ensemble", linestyle='-', color='red')

    # Threshold 시각화
    plt.axhline(ensemble_threshold, color='black', linestyle='--', label="Threshold")

    # 그래프 제목과 축
    graph_file_name = "./result/result-{}.png".format(file_time)
    plt.title("Comparison of Model Scores Over Time")
    plt.xlabel("Records")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig(graph_file_name, bbox_inches='tight')

    # Ensure flattened arrays for MSE and MAE
    ensemble_scores_flattened = ensemble_scores_flattened.flatten()
    reconstructed_data_flattened = np.mean(reconstructed_data, axis=1)
    reconstructed_data_flattened = reconstructed_data_flattened.flatten()


    # KMeans for Silhouette Score Calculation
    # print("KMeans clustering started:", datetime.now())
    # kmeans = KMeans(n_clusters=2, random_state=42)
    # kmeans_labels = kmeans.fit_predict(ensemble_scores_flattened.reshape(-1, 1))
    # silhouette = calculate_silhouette(ensemble_scores_flattened.reshape(-1, 1), kmeans_labels)
    silhouette = silhouette_score(ensemble_scores_flattened, outlier_labels)
    

    # print("KMeans clustering ended:", datetime.now())
    mse = mean_squared_error(ensemble_scores_flattened, reconstructed_data_flattened)
    mae = mean_absolute_error(ensemble_scores_flattened, reconstructed_data_flattened)

    # Print Results
    print("Silhouette Score:", silhouette)
    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("Threshold:", ensemble_threshold)
    print("Anomalies Detected:", np.sum(anomalies_flattened))
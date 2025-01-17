# ILAGE 

2025 32th KCS Conference on Semiconductors 발표 논문
- Implementation of a Python-Based QLC Cell Defect Prediction Model and Study on Reliability Improvement

## Requirements
제공되는 파이썬 코드를 실행하기 위해서는 사전에 환경 설정이 필요합니다.
- Python 3.10 (venv 환경으로 구성)
- Optional : CUDA/CUDNN, LSTM 모델 학습 속도 향상을 위해서 사용합니다.

## How to run
### Windows
- 파이썬 설치 필요
- 파이썬 실행 가능(python/python3 Alias에 따라 실행 명령어 다름)
```
python3 -m venv venv
venv/Scripts/activate.bat
pip install -r ./requirements.txt
python3 ./data_processing.py
python3 ./data_aug.py
python3 ./train_ILAGE.py
```

### Ubuntu
- Python-3.10, Python-3.10-venv 사전 apt-get 으로 설치 필요
- 파이썬 실행 가능(python/python3 Alias에 따라 실행 명령어 다름)
```
python3 -m venv venv
source venv/bin/activate
pip install -r ./requirements.txt
python3 ./data_processing.py
python3 ./data_aug.py
python3 ./train_ILAGE.py
```

## Hyperparameter Setting
| Model         | Parameter           | Value                            |
|---------------|---------------------|----------------------------------|
| Isolation Forest | n_estimators      | 100                             |
|               | contamination       | 0.05                             |
|               | random_state        | 42                               |
| LSTM-AE       | Input Shape         | (3, 9)                           |
|               | LSTM Layers(Encoder)| [64, 32]                         |
|               | LSTM Layers(Decoder)| [32, 64]                         |
|               | Epochs              | 100                              |
|               | Batch Size          | 32                               |
|               | Loss                | MSE                              |
| GAN           | Gen. Layers         | [64, 27] (tanh, sigmoid)         |
|               | Disc. Layers        | [64, 1] (tanh, sigmoid)          |
|               | Epochs              | 100                              |
|               | Batch Size          | 32                               |
|               | Loss                | Binary Crossentropy              |
| Ensemble      | Weight              | [0.4, 0.3, 0.3]                  |
|               | Threshold           | 95%                              |


## Result


<!--
## BibTeX

```
@misc{}
```
-->
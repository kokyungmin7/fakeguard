# FakeGuard

## 프로젝트 구조

```
your_submission/                 # 제출 ZIP 최상위 디렉터리
├── model/                       # 필수요건
│   └── model.pt                 # 필수요건 – 최종 추론에 사용하는 단일 모델 weight 1개
│
├── src/                         # 희망요건 – 모델/데이터/유틸 모듈 분리
│   ├── models.py                # 희망요건 – 모델 정의
│   ├── dataset.py               # 희망요건 – 데이터 로더/전처리
│   └── utils.py                 # 희망요건 – 공통 유틸 함수
│
├── config/                      # 필수요건
│   └── config.yaml              # 필수요건 – 하이퍼파라미터, 경로 키, 모델명 등
│
├── env/                         # 필수요건
│   ├── Dockerfile               # 필수요건 – 제출 Docker 이미지 재현용
│   ├── requirements.txt         # 필수요건 – 추가 Python 라이브러리 목록
│   └── environment.yml          # 희망요건 – 로컬/연구용 conda 환경 정의
│
├── train_data/                  # 필수요건
│   └── 학습 데이터              # 필수요건 – 학습 데이터(재현용, 데이터 전수), 출처 및 라이선스
│
├── test_data/                   # 필수요건
│   └── 평가 데이터              # 필수요건 – 경진대회 제공 평가 데이터셋
│
├── train.py                     # 필수요건 – config, train_data 기반 학습 코드
├── eval.py                      # 희망요건 – 내부 검증용 평가 코드(ROC-AUC 등)
├── inference.py                 # 필수요건 – 채점용 추론 엔트리 포인트
└── README.md                    # 필수요건 – 전체 구조/환경/실행 방법 설명
```

## 환경 설정

### Conda 환경 사용 (희망요건)

```bash
conda env create -f env/environment.yml
conda activate fakeguard
```

### pip 사용

```bash
pip install -r env/requirements.txt
```

## 실행 방법

### 학습

```bash
python train.py
```

학습은 `config/config.yaml`의 설정과 `train_data/`의 데이터를 기반으로 수행됩니다.

### 평가

```bash
python eval.py
```

내부 검증용 평가 코드로 ROC-AUC 등의 평가 지표를 계산합니다.

### 추론

```bash
python inference.py
```

채점용 추론 엔트리 포인트입니다. `model/model.pt`의 모델을 로드하여 추론을 수행합니다.

## Docker 사용

### 이미지 빌드

```bash
docker build -f env/Dockerfile -t fakeguard .
```

### 컨테이너 실행

```bash
docker run fakeguard
```

## 설정 파일

`config/config.yaml`에서 하이퍼파라미터, 경로, 모델명 등을 설정할 수 있습니다.

## 데이터

- `train_data/`: 학습 데이터 디렉토리 (재현용, 데이터 전수), 출처 및 라이선스 포함
- `test_data/`: 경진대회 제공 평가 데이터셋 디렉토리

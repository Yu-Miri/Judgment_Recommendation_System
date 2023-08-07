# **Judgment Recommendataion System**
# 프로젝트 개요

- **기간** : 2023.02.16 ~ 2023.03.09
- **프로젝트 진행 인원** : 3명
- **주요 업무 및 상세 역할**
    - 프로젝트 **주제 선정 및 계획 수립**
    - 판결문 **크롤링**
    - 불용어 사전 제작 및 **데이터 전처리**
    - 사전 학습된 SBERT 모델의 데이터셋 구조를 파악하여 **Custom Dataset 생성**
    - 구축한 데이터셋을 학습시키면서 KoBERTa **모델의 성능 평가 및 모델 핸들링**
    - 프로젝트 **발표자료 작성**
- **사용언어 및 개발환경** : Google colab Pro+, Python, BeautifulSoup, Mac OS, Pytorch, Tensorflow, Pandas, Re, Hugging Face, VSCode, Slack

## 문제 정의

- 일상에서 법률적인 대처가 필요한 상황이 발생하는 경우가 많지만 법률과 거리가 먼 사람들은 제대로 대응하기에 어려움이 있으며, 변호사 상담을 위한 비용 측면에서 합리적인 비용인지에 대한 판단이 흐려지는 것에 더하여, 변호사와의 접근성이 낮아 시간적으로 소요가 많이 된다. 이에 따라 빠르게 대처해야 하는 법률적 문제 상황에서 혼란을 겪게 된다.
- 아래의 사진과 같이 합리적인 변호사 상담 비용과 변호사와의 접근성을 높인 변호사 상담 앱 ‘로톡’의 매출은 2023.02 기준 전년비 2배 상승하였으며, 앱의 방문자 또한 증가하는 추세이다. 이를 통해 법률적 자문을 원하는 소비자들이 늘어나고 있음을 알 수 있다.
  
<img width="620" alt="스크린샷 2023-07-27 오후 12 48 16" src="https://github.com/Yu-Miri/Judgment_Recommendation_System/assets/121469490/620852e7-9e21-4cc4-a038-e8fa92cede0f">

## 해결 방안

### <프로젝트 목적>

- NLP를 통해 판결문의 법률 용어를 학습에 방해가 되는 불용어를 제거하거나 이외의 토큰화, 품사 태깅 등의 전처리를 적용하고, 법률 용어에 Pretrain된 Model을 Fine Tuning하여 모델의 성능을 개선하여 해당 상황과 유사한 판결문을 예측할 수 있도록 모델링 경험

### <프로젝트 내용>

- 팀 프로젝트를 통해 소비자가 법률적 문제 상황에 대하여 빠르게 대처할 수 있도록 자신의 문제 상황을 정해진 형식에 맞게 요약하여 제출하면 딥러닝을 통해 해당 문장을 분석하여 이전에 존재했던 판결문 중에서 가장 유사한 판결문을 제공하고, 이에 더하여 법률 관련 전문가의 보조 수단으로도 유사한 판결문을 제공

## 데이터 설명

- **출처** : CaseNote
- **Crawling Data** : 8가지 주제의 판결문 2,000개
- **주제** : 마약, 명예훼손, 모욕, 방화, 사기, 성추행, 의료사고, 이혼

<img width="642" alt="스크린샷 2023-07-27 오후 12 49 09" src="https://github.com/Yu-Miri/Judgment_Recommendation_System/assets/121469490/a57a707d-59d5-4542-ade8-be383b0738db">

- **Train Dataset**
    - KLUE NLI Dataset(570,000) + KLUE STS Dataset(5,000) + Custom STS Dataset (125,000)
    - KLUE(Korean Language Understanding Evaluation) Dataset : 대화 상태 추적, 기계 독해, 의미적 텍스트 유사성, 관계 추출 등 한국어 자연어 이해(NLI) 작업을 위해 구성된 데이터셋
      
    <img width="796" alt="스크린샷 2023-07-27 오후 12 49 48" src="https://github.com/Yu-Miri/Judgment_Recommendation_System/assets/121469490/2dafba91-e99e-4fc8-b18a-f62a05935b4d">

    
    - 두 문장이 유사한지 파악하여 3개의 Class로 분류하며, Classification Accuracy를 평가 지표로 사용한다.
  
    <img width="795" alt="스크린샷 2023-07-27 오후 12 50 15" src="https://github.com/Yu-Miri/Judgment_Recommendation_System/assets/121469490/0748ec9b-055b-44af-a5a9-2617d029bacd">

    - 두 문장 간의 유사도를 0에서 5까지 평균 실수 값으로 Labeling되었으며, 말뭉치 간의 유사성 파악을 위해 피어슨 상관계수와 F1-score를 평가 지표로 사용한다.
  
    <img width="802" alt="스크린샷 2023-07-27 오후 12 50 45" src="https://github.com/Yu-Miri/Judgment_Recommendation_System/assets/121469490/deedbde4-24a8-40d2-b33a-bfadf774430d">

    - STS Dataset을 Custom한 것으로, 하나의 판결문 내에서 문장 단위의 유사도를 학습하기 위해 구두점을 기준으로 split해서 판결문 하나의 문장에 대해 나머지 문장들을 각각 하나의 열로 묶어 Cosine Similarity Loss를 채택하여 두 문장 간의 문장 임베딩 벡터 Score를 Normalization

## 데이터 전처리

- **크롤링한 판결문 1차 전처리 결과**

    <img width="797" alt="스크린샷 2023-07-27 오후 12 50 58" src="https://github.com/Yu-Miri/Judgment_Recommendation_System/assets/121469490/18c51f1e-8b94-4194-8dba-190e82f07a2b">

    - 한글, 구두점, 쉼표, 공백을 제외한 문자 제거 : 특수문자, 숫자 등 의미 없는 문자 존재
    - 공백 치환 : 쉼표를 공백으로 치환 후 다수 개의 공백을 하나의 공백으로 치환
    - 노이즈, 불용어 제거 : ‘가’,’나’,’다’ 등의 글머리 및 ‘사건 개요’ 등의 형식적 단어를 선정
      
- **크롤링한 판결문 2차 전처리 결과**

    <img width="797" alt="스크린샷 2023-07-27 오후 12 51 30" src="https://github.com/Yu-Miri/Judgment_Recommendation_System/assets/121469490/afdcacc4-945e-4fce-9c94-8249072992e5">

    - 1차 전처리로 의미없이 남게 되는 한 글자 또는 두 글자에 대하여 추가 불용어 사전 제작

## 선정 모델

- **Sentence BERT**
    - 선정 이유 : BERT 모델은 주로 단어 수준의 임베딩을 학습하므로, SBERT와 비교했을 때 상대적으로 문장 간의 유사도를 구하는 연산 속도가 느리고, 장문의 유사도 판별 성능이 낮지만, 두 문장 간의 의미적 유사성을 측정하기 위해 사용되는 SBERT는 BERT 모델을 문장 수준으로 확장한 원샷 모델이기 때문에 연산 속도가 빠르며, 장문의 유사도 판별 성능이 높은 모델
    <img width="236" alt="스크린샷 2023-07-27 오후 12 51 58" src="https://github.com/Yu-Miri/Judgment_Recommendation_System/assets/121469490/50feabb3-1e2f-40dc-b8b7-ef8e08030f8d">
    
    - **스피어만 상관계수**
        - 두 변수 간의 상관관계를 측정하기 위해 사용되는 통계적인 지표로서, -1에서 1 사이의 값을 가진다.
        - 두 변수 간의 순위 순서를 기반으로 강도와 방향성을 측정하며, 선형적인 상관 관계를 나타내는 것이 아닌 한 변수가 증가하는 경우에 다른 변수가 증가하는지 감소하는지에 대한 관계를 나타낸다.
    - 왼쪽의 지표에서 보이는 것과 같이 BERT의 임베딩 Score는 46.35이지만, SBERT의 임베딩 Score는 77~88의 Score로 SBERT의 문장 벡터 구성의 성능이 월등하다는 것을 알 수 있다.


    - **Task[Classifier vs. Regression]**

        <img width="620" alt="스크린샷 2023-07-27 오후 12 52 44" src="https://github.com/Yu-Miri/Judgment_Recommendation_System/assets/121469490/468e92e2-641e-4ed0-a033-c7dc098430b1">

        - **Classifier** : 두 문장이 BERT 모델을 통해 각각 문장의 의미를 담고 있는 u, v 임베딩 벡터를 추출하며, u와 v의 차이에 절대값을 취하여 Softmax로 Classifier
        - **Regression** : 두 문장이 BERT 모델을 통해 각각 문장의 의미를 담고 있는 u, v 임베딩 벡터를 추출하며, 벡터 간의 방향과 크기를 기반으로 유사도를 구하는 cosine-similarity로 u와 v 간의 유사도를 -1에서 1 사이의 값으로 Regression

## 모델링 & 성능

- Pretrained Model Train
    
    
    | Model | Transformer | Train Dataset |
    | --- | --- | --- |
    | SBERT-1 | KoBERTa | Custom STS Dataset(125,000) |
    | SBERT-2 | KLUE RoBERTa | Custom STS Dataset(125,000) |
    | SBERT-3 | KoBERTa | Total Dataset(700,000) |
    - KoBERTa : 한국어 문장 임베딩을 위한 한국어 BERT 모델
    - KLUE RoBERTa : KLUE benchmark에서 성능을 향상시키기 위한 한국어 문장 임베딩 BERT 모델
    - KoBERTa와 KLUE RoBERTa를 각각 Custom STS Dataset으로 학습시키고, 더 높은 성능을 가지는 모델에 KLUE NLI Dataset과 KLUE STS Dataset을 추가로 학습시켜 모델의 성능을 개선하였다.

- 모델의 성능 비교
    - 유사한 문장의 성능
      
        <img width="778" alt="스크린샷 2023-07-27 오후 12 53 06" src="https://github.com/Yu-Miri/Judgment_Recommendation_System/assets/121469490/d15be536-5393-472a-a312-ce454f7a274f">

    - 유사하지 않은 문장의 성능
      
        <img width="779" alt="스크린샷 2023-07-27 오후 12 53 22" src="https://github.com/Yu-Miri/Judgment_Recommendation_System/assets/121469490/7d77535d-6b44-41d6-8db6-b3eb94353dbe">

        
    - Custom STS Dataset으로 학습한 두 개의 Pretrained Model의 성능을 각각 비교해 보았을 때 유사한 문장의 경우 SBERT-2 모델의 성능이 더 좋았으며, 유사하지 않은 문장의 경우 SBERT-1 모델의 성능이 더 좋았다.
    - 유사한 문장의 성능에서 유사도가 상대적으로 높게 나온 SBERT-2 모델은 유사하지 않은 문장에도 유사도가 높게 나왔지만, SBERT-1 모델은 유사한 문장의 성능이 상대적으로 낮아도 유사하지 않은 문장의 성능에서 유사도가 현저히 낮게 나왔기에 SBERT-1 모델의 성능이 더 좋다고 판단하였다.
    - 판단에 근거하여 SBERT-1 모델에 추가로 전체 데이터셋(700,000)을 학습시킨 SBERT-3 모델의 성능은 유사한 문장의 유사도가 전보다 약 0.01 하락했음에도, 유사하지 않은 문장의 유사도가 약 0.15 하락함에 따라 모델의 성능이 개선된 것을 볼 수 있었다.

## 개선사항 &  Inference

- 개선사항
    - 크롤링 데이터 : 사이트 또는 판결문의 작성 구조가 전부 달라서 크롤링 데이터의 제한적 사용
    - Sentence BERT : 모델의 output에 판결문을 판단하는 알고리즘을 개선
    - 최종 모델의 선정 : 각각 학습시킨 3가지 모델을 Voting하여 모델을 선정하는 idea 적용

- Inference
<img width="820" alt="스크린샷 2023-07-27 오후 12 53 44" src="https://github.com/Yu-Miri/Judgment_Recommendation_System/assets/121469490/4256bf14-c5b3-4822-ae22-ca6cd0bd1071">

  
## Installation

### Requirements

- Python==3.8

      git clone https://github.com/Yu-Miri/Judgment_Recommendation_System.git
      pip install -U sentence-transformers
  
### Preparing the Dataset
https://drive.google.com/drive/folders/1IvaShVNQsqIJONhFaalnhDGhNeuaOOeO?usp=sharing

### train

      python train.py

### Inference

      import inference from inference

      data = pd.read_csv('dataset/raw_data/의료사고결과.csv') # inference할 판결문 데이터 불러오기
      NUM_SAMPLES = 200 # 사용할 판결문 개수 지정
      NUMBER = 10 # Inference 할 판결문 번호

      inference(data, NUM_SAMPLES, NUMBER)

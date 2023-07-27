import re
import pandas as pd
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from make_embedding_vector import make_embedding
from data_util import cal_cosine_inference

def inference(data, NUM_SAMPLES, NUMBER):
    # 전처리한 판결문 데이터를 임베딩 벡터로 변환하여 dataset/train_data/Custom_STS저장 후
    df = make_embedding(data)
    df = df[:NUM_SAMPLES]

    # model = SentenceTransformer('kor_multi_klue-roberta-base-2023-03-06_15-01-09')
    model_src = SentenceTransformer('jhgan/ko-sroberta-multitask')

    similarity_list = df['similarity_list'].tolist()

    test_sentence = df.iloc[NUMBER]['sen_list']
    # test_emb = model.encode(test_sentence[0])
    test_emb_src = model_src.encode(test_sentence[0])

    # cosine_list = []
    cosine_list_src = []

    for similarity in similarity_list:
        # cosine_list.append(cal_cosine_inference(test_emb, similarity))
        cosine_list_src.append(cal_cosine_inference(test_emb_src, similarity))

    # df['top5'] = cosine_list

    # df['mean'] = df['top5'].apply(lambda x : sum(x)/len(x))
    # df['max'] = df['top5'].apply(lambda x : max(x))
    # df['rank'] = df['mean'].rank() + df['max'].rank()

    # idx = np.argmax(df['rank'])
    # pan = df['이유'][idx]

    df['top5_s'] = cosine_list_src

    df['mean_src'] = df['top5_s'].apply(lambda x : sum(x)/len(x))
    df['max_src'] = df['top5_s'].apply(lambda x : max(x))
    df['rank_src'] = df['mean_src'].rank() + df['max_src'].rank()

    idx_src = np.argmax(df['rank_src'])
    pan_src = df['sen_list'][idx_src]

    simi = df.iloc[idx]['top5']

    display(f'의뢰사연: {test_sentence[0]}')
    print(f'유사한 판결문 번호: {idx_src}')
    print(f'유사한 판결문 유사성: {simi}')
    print(f'유사한 판결문 원본: {pan_src}')
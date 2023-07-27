### Preparing a preprocessed dataframe
from sentence_transformers import SentenceTransformer
import pandas as pd
from df_raw_preprocessing import df_preprocess, preprocess

def make_embedding(data):
    df = pd.DataFrame(data['이유'])
    df = df.rename(columns={'이유': 'verdict'})
    df_new = df_preprocess(df)

    # 한국어 문장 임베딩 모델을 불러와서 문장 Embedding한다.
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    NUM_SAMPLES = 200
    from tqdm import tqdm
    
    verdict = []
    sen_emb_vec = []
    
    for i in tqdm(range(NUM_SAMPLES)):
        verdict.append(df_new['sentence'][i])
        sen_emb_vec.append(model.encode(df_new['sentence'][i]))

    df['sen_list'] = verdict
    df['similarity_list'] = sen_emb_vec
    
    save_dir = 'dataset/Custom_STS/'
    # split한 한국어 문장의 데이터 프레임을 한 줄씩 불러와 임베딩하여 csv 파일으로 저장한다.
    df.to_pickle(save_dir+'custom_sts_dataset_embedding.csv')

    return df
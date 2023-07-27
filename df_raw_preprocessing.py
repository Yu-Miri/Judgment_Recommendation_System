import re
import pandas as pd
from tqdm import tqdm

def preprocess(sen):
    #한글과 구두점을 제외하고 '가,나,다,라,마,바,사,아' 문자와 특수문자, 숫자를 제거한다.
    sen = re.sub(r'\n', r' ', sen)
    sen = re.sub(r'[0-9]+건.+? ', r' ', sen)
    sen = re.sub(r'[0-9].', r' ', sen)
    sen = re.sub(r'[0-9]+\.', r' ', sen)
    sen = re.sub(r'\([^)]*\)', r'', sen)
    sen = re.sub(r' [가나다라마바사아자차]\s?\.', r'', sen)
    sen = re.sub(r'[가-힣]○○', r'', sen)
    sen = re.sub(r'[^ㄱ-ㅎ가-힣\.]+', r' ', sen)
    sen = re.sub(r' [가나다라마바사아자차] ', r' ', sen)
    
    # 노이즈, 불용어를 제거한다.
    for stop in stopwords:
        sen = re.sub(stop, r' ', sen)

    #단어와 구두점, 쉼표 사이에 공백을 제거한다.
    sen = re.sub(r'([\.])', r' \1', sen)

    #다수 개 공백을 하나의 공백으로 치환한다.
    sen = re.sub(r"\s+", " ", sen)

    s_split = sen.split('.')

    # 문장을 온점으로 split하여 길이가 4 이상인 글자만 반환한다.
    return_s = [s for s in s_split if len(s)>3]

    return '.'.join(return_s)

def df_preprocess(df):
    '''
    input (df) = 판결문 데이터프레임
    '''
    # 불용어 사전을 준비한다.
    stopword = open('/law_stopwords.txt', 'rt').read()
    stopwords = stopword.split('\n')

    sentence = df['verdict']
    tqdm.pandas()

    # 판결문을 전처리하여 데이터 프레임에 추가한다.
    df['pre_verdict'] = sentence.apply(lambda x : preprocess(x))

    # 전처리된 하나의 판결문에 대해서 온점을 기준으로 split 하여 하나의 리스트로 데이터 프레임에 추가한다.
    df['pre_split'] = df['pre_verdict'].apply(lambda x : x.split('.'))

    pan_num = []
    sentence = []
    for i, pre in enumerate(df['pre_split']): #해당 index, pre_split의 한 문장
        pan_num.extend([i] * len(pre))
        sentence.extend(pre)
    df_new = pd.DataFrame(zip(pan_num, sentence), columns = ('pan_num', 'sentence'))
    return df_new
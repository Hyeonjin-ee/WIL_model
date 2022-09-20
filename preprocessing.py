import boto3 # S3버킷과 연동
import datetime
import time 
import pandas as pd
import re
from tqdm import tqdm
from datetime import date
from sklearn.model_selection import train_test_split
import os

def removing_non_korean(df):
    for idx, row in tqdm(df.iterrows(), desc='removing_non_korean', total=len(df)):
        new_doc = re.sub('[^가-힣]', '', row['document']).strip()
        df.loc[idx, 'document'] = new_doc
    return df

def generate_preprocessed():
    today = date.today().strftime("%m%d")
    data_path = "~/data/model_test/WIL_model/data"
    s3 = boto3.client('s3')
    #s3.download_file('wilcrawling', f'dataset-{today}.txt', f'./data/dataset_{today}.txt')

    data = pd.read_csv(f'../data/dataset_{today}.txt', 
                         delimiter='\t',
                         encoding='utf-8') 

    # null 제거
    data.dropna(inplace=True)

    # 중복 제거
    data.drop_duplicates(subset=['document'], inplace=True, ignore_index=True)

    # 한국어 외 제외
    data = removing_non_korean(data)

    # train : test = 3 : 1
    train, test = train_test_split(data, test_size=0.25, random_state=25)

    # 파일 저장
    train.to_csv(f'{data_path}/train.txt', index=False, sep='\t')
    test.to_csv(f'{data_path}/test.txt', index=False, sep='\t')


if __name__=="__main__":
    generate_preprocessed()
    


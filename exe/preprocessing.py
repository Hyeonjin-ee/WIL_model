import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

BASE_DIR = "/root/data/model_test/WIL_model"

def train_test():
    
    # WIL_model/data 에 있는 dataset_0~6까지 concat 해서 csv(sep=/t) 되어있는 파일로 빼기
    total = []
    target = "/root/data/model_test/WIL_model/data/txtdata/"
    for file in os.listdir(target):
        total.append(pd.read_csv(target+file, delimiter='\t' , encoding='utf-8'))

    concat_df = pd.concat(total[0:7])

        
    # train : test = 3 : 1
    train, test = train_test_split(concat_df, test_size=0.25, random_state=25)
    

    # 파일 저장
    train.to_csv(f'{BASE_DIR}/data/train.txt', index=False, sep='\t')
    test.to_csv(f'{BASE_DIR}/data/test.txt', index=False, sep='\t')


if __name__=="__main__":
    train_test()
    


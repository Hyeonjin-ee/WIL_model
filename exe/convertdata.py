import pandas as pd
import datetime
import time 
from datetime import date

today = datetime.datetime.today().weekday()
BASE_DIR = "/root/data/model_test/WIL_model"

def convert_data():
    df = pd.read_json(f'{BASE_DIR}/data/dataset_{today}.json')
    df.columns=["label", "text"]
    df.to_csv(f'{BASE_DIR}/data/txtdata/dataset_{today}.txt', sep='\t', index = False)
    
if __name__=="__main__":
    convert_data()
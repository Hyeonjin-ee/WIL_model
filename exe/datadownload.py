from pathlib import Path  
import environ
import os
import datetime
from datetime import date
import boto3

BASE_DIR = "/root/data/model_test/WIL_model"
today = datetime.datetime.today().weekday()


env = environ.Env(
    # set casting, default value
    DEBUG=(bool, False)
)

# Take environment variables from .env file
environ.Env.read_env(
    os.path.join(BASE_DIR, '.env')
)


def download_data():
    file_name = str(datetime.datetime.now())[:-16]
    client = boto3.client('s3', aws_access_key_id=env('AWS_ACCESS_KEY_ID'), aws_secret_access_key=env('AWS_SECRET_ACCESS_KEY'))
    response = client.list_buckets()

    client.download_file('wil-airflow', f'{file_name}.json', f'{BASE_DIR}/data/dataset_{today}.json')
    
    
if __name__=="__main__":
    download_data()
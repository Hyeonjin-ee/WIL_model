from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.contrib.operators.ssh_operator import SSHOperator
from airflow.contrib.hooks.ssh_hook import SSHHook
from datetime import datetime, timedelta
import environ
import os
from pathlib import Path  
from botocore.config import Config
import boto3, json

BASE_DIR = '/home/airflow/airflow_docker/example/dags' 

env = environ.Env(
    # set casting, default value
    DEBUG=(bool, False)
)

environ.Env.read_env(
    os.path.join(BASE_DIR, '.env')
)

# argument 작성 
args = {
    'owner': 'jonghyeok',
    'retries': 3,
    'retry_delay': timedelta(minutes=1), 
}

# crawling 및 전처리를 위한 AWS의 lambda 함수 불러오기
def lambda_crawling(ds,**kwargs,): # 
    my_config = Config(
        region_name = 'ap-northeast-2',
        signature_version = 'v4',
        retries = { 'max_attempts': 10, 'mode': 'standard'}
        ) 

    event = dict()

    client = boto3.client(
        'lambda',
        aws_access_key_id=env('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=env('AWS_SECRET_ACCESS_KEY'),
        config=my_config,
        )
    
    response = client.invoke(FunctionName='airlfow-lambda' , Payload=json.dumps(event))
    print(response) 
      
# DAG 작성
dag = DAG(
    dag_id='lambda_crawling',
    default_args=args, 
    description="(daily)AWS Lambda를 통해 S3에 데이터 수집, 전처리 및 GPU_server에 전달 후 파일 변환", 
    start_date=datetime(2022, 9, 26), 
    schedule_interval = '40 14 * * *', # 매일 23:40분에 동작하도록 시간 설정
    )

# DAG Task 작성
# 1. crawling
naver_crawling = PythonOperator(
    task_id="lambda_crawling", 
    python_callable=lambda_crawling, 
    provide_context=True,
    dag=dag
)

bash_command="""
        {% for task in dag.task_ids %}
            echo "{{ task }}"
            echo "{{ ti.xcom_pull(task) }}"
        {% endfor %}
    """

# 2. Show result
show_data = BashOperator(
    task_id='show_result',
    bash_command=bash_command,
)

# GPU_server와 ssh telnet 연결을 위한 hook 설정
hook = SSHHook(
    username='root',
    remote_host='deep.kbig.kr',
    port= env('GPUPORT'),
    password=env('GPUPWD'),
)

# 3. Crawling한 데이터 S3에서 Dawnload
S3_download_data = SSHOperator(
    task_id='S3_download_data',
    ssh_hook=hook,
    command="cd data/model_test/WIL_model/script && ./data_load.sh*",
)

# 3. json -> txt 파일 형태 변환
jsonToText = SSHOperator(
    task_id='jsonToText',
    ssh_hook=hook,
    command="cd data/model_test/WIL_model/script && ./convert.sh*",
)

# tasks 의존관계 작성
naver_crawling >> show_data
show_data >> S3_download_data
S3_download_data >> jsonToText
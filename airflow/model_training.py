from airflow import DAG
from airflow.contrib.operators.ssh_operator import SSHOperator
from airflow.contrib.hooks.ssh_hook import SSHHook
from datetime import datetime, timedelta
import environ
import os

BASE_DIR = '/home/airflow/airflow_docker/example/dags/' 

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
    'retry_delay': timedelta(minutes=3),
}

# DAG 작성
dag = DAG(
    dag_id='model_update', 
    default_args=args, 
    description="(weekly) 학습 데이터 전처리 및 모델 학습 후 model에 git push와 pull을 활용해 업데이트", 
    start_date=datetime(2022, 9, 26), 
    schedule_interval = '0 0 * * 0', # 매주 일요일 자정에 실행
    )

# GPU_server와 ssh telnet 연결을 위한 hook 설정
hook_model = SSHHook(
    username='root',
    remote_host='deep.kbig.kr',
    port= env('GPUPORT'),
    password=env('GPUPWD'),
)

# DAG Task 작성
# 1. 학습을 위한 데이터 processing (concat + trainset/testset split)
data_processing = SSHOperator(
    task_id='data_processing',
    ssh_hook=hook_model,
    command="cd data/model_test/WIL_model/script && ./pre.sh*",
    dag=dag
)

# 2. model 학습
model_training = SSHOperator(
    task_id='model_training',
    ssh_hook=hook_model,
    command="cd data/model_test/WIL_model/script && ./train.sh*",
)

# 3. 학습된 결과를 git에 push
git_push = SSHOperator(
    task_id='git_push',
    ssh_hook=hook_model,
    command="cd data/model_test/WIL_model/script && ./git-push.sh*",
)

# Serving_model의 ssh telnet 연결을 위한 hook 설정
hook_serving_model = SSHHook(
    remote_host=env('REMOTE_HOST'),
    port=env('PORT'),
    username=env('USERNAME'),
    password=env('PWD'),
)

# 4. 학습된 결과를  Serving_server로 pull
git_pull = SSHOperator(
    task_id='git_pull',
    ssh_hook=hook_serving_model,
    command="cd /home/ubuntu/WIL_model/script && ./git-pull.sh*",
    dag = dag,
)

# tasks 의존관계 작성
data_processing >> model_training
model_training >> git_push
git_push >> git_pull
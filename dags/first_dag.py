from airflow import DAG
from airflow.operators.python import PythonOperator

from datetime import datetime, timedelta 
import pandas as pd
from pathlib import Path
import sys

root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))

from dags import PROJECT_FOLDER
from dags.src.data import DATASET_NAME
from dags.src.data.download import download_and_store_data
from dags.src.data.data_stats import data_stats
from dags.src.data.logger_info import setup_logger
from dags.src.data.data_split import split_data
from dags.src.data.pre_process import conver_to_list
from dags.src.data.tokenise_data import tokenise_json_data
from dags.src.data.gcloud_upload import upload_files
from dags.src.model.generate_tokeniser import genereate_tokenizer
# split_data(PROJECT_FOLDER)


logger  = setup_logger(root_dir, 'download_data')
token_logger  = setup_logger(root_dir, 'tokenise_data')

dag_owner = 'Harshan_Ganugula'

default_args = {'owner': dag_owner,
        'depends_on_past': False,
        'retries': 1
        }

import subprocess
def list_installed_packages():
    result = subprocess.run(['pip', 'list'], stdout=subprocess.PIPE)
    packages = result.stdout.decode('utf-8')
    print(packages)

dag = DAG(dag_id='first_dag_run',
        default_args=default_args,
        description='This dag loads the data and pre-process data.',
        start_date=datetime(2023,1,1),
        schedule_interval='@daily',
        catchup=False,
        tags=['data', 'pre-processing'])

download_data_dag = PythonOperator(
    task_id = 'download_and_store_data',
    python_callable = download_and_store_data,
    op_kwargs = {'DATASET_NAME': DATASET_NAME, 'logger': logger, 'root_dir': PROJECT_FOLDER},
    dag = dag
)

list_packages_task = PythonOperator(
    task_id='list_installed_packages',
    python_callable=list_installed_packages,
    dag=dag,
)

split_data_dag = PythonOperator(
    task_id = 'split_data',
    python_callable = split_data,
    op_kwargs = {'PROJECT_FOLDER': PROJECT_FOLDER, 'logger': setup_logger(root_dir, 'data_split')},
    dag = dag
)

convert_train_list_dag = PythonOperator(
    task_id = 'Convert_TRAIN_List',
    python_callable = conver_to_list,
    execution_timeout=timedelta(minutes=10),  # set the timeout to 30 minutes
    op_kwargs = {'PROJECT_FOLDER': PROJECT_FOLDER, "FILE": "train", 'logger': setup_logger(root_dir, 'convert_train_list')},
    dag = dag
)

convert_test_list_dag = PythonOperator(
    task_id = 'Convert_TEST_List',
    python_callable = conver_to_list,
    op_kwargs = {'PROJECT_FOLDER': PROJECT_FOLDER, "FILE": "test",'logger': setup_logger(root_dir, 'convert_test_list')},
    dag = dag
)

generate_tokeniser_dag = PythonOperator(
    task_id = 'Generate_Tokeniser',
    python_callable = genereate_tokenizer,
    op_kwargs = {'PROJECT_FOLDER': PROJECT_FOLDER, 'logger': setup_logger(root_dir, 'genereate_tokenizer')},
    dag = dag
)
generate_token_data_dag = PythonOperator(
    task_id = 'TokenData',
    python_callable = tokenise_json_data,
    op_kwargs = {'PROJECT_FOLDER': PROJECT_FOLDER, 'logger': setup_logger(root_dir, 'generate_token_data')},
    dag = dag
)

generate_data_stats = PythonOperator(
    task_id = 'DataStats',
    python_callable = data_stats,
    op_kwargs = {'PROJECT_FOLDER': PROJECT_FOLDER, 'LOGGER': setup_logger(root_dir, 'data_stats')},
    dag = dag
)

gcloud_upload = PythonOperator(
    task_id = 'gcloud_upload',
    python_callable = upload_files,
    op_kwargs = {'PROJECT_FOLDER': PROJECT_FOLDER, 'BUCKET_NAME': 'finer_data_bk', 'logger': setup_logger(root_dir, 'generate_token_data')},
    dag = dag
)

download_data_dag >> list_packages_task >> split_data_dag >> [convert_test_list_dag, convert_train_list_dag] # type: ignore
convert_train_list_dag >> generate_tokeniser_dag # type: ignore
generate_tokeniser_dag >> generate_token_data_dag # type: ignore
generate_token_data_dag >> generate_data_stats >> gcloud_upload # type: ignore

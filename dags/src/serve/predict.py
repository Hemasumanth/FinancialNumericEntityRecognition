import os
import re
import json
import pickle
import joblib
import time
import shutil
import numpy as np
from pathlib import Path
from google.cloud import storage
import tensorflow as tf
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from datetime import datetime
from google.api_core.exceptions import NotFound
from google.cloud import storage, logging, bigquery
from google.cloud.bigquery import SchemaField
from google.oauth2 import service_account
from google.logging.type import log_severity_pb2 as severity

from tensorflow.keras.preprocessing.sequence import pad_sequences

load_dotenv()

# Set up Google Cloud logging
service_account_file = 'finerteam8-00bf2670c240.json'
credentials = service_account.Credentials.from_service_account_file(service_account_file)
client = logging.Client(credentials=credentials)
logger = client.logger('Serving_pipeline')

bq_client = bigquery.Client(credentials=credentials)
table_id = os.environ['BIGQUERY_TABLE_ID']

# Now you can use os.getenv to access your environment variables
app = Flask(__name__)

def get_table_schema():
    """Build the table schema for the output table
    
    Returns:
        List: List of `SchemaField` objects"""
    return [
        SchemaField("Text", "STRING", mode="REQUIRED"),
        SchemaField("length", "INTEGER", mode="REQUIRED"),
        SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
        SchemaField("oov_words", "INTEGER", mode="REQUIRED"),
        SchemaField("warning_flag", "INTEGER", mode="NULLABLE"),
        SchemaField("latency", "FLOAT", mode="NULLABLE"),
        SchemaField("predictions", "STRING", mode="REPEATED")
    ]

def create_table_if_not_exists(client, table_id, schema):
    """Create a BigQuery table if it doesn't exist
    
    Args:
        client (bigquery.client.Client): A BigQuery Client
        table_id (str): The ID of the table to create
        schema (List): List of `SchemaField` objects
        
    Returns:
        None"""
    try:
        client.get_table(table_id)  # Make an API request.
        print("Table {} already exists.".format(table_id))
    except NotFound:
        print("Table {} is not found. Creating table...".format(table_id))
        table = bigquery.Table(table_id, schema=schema)
        client.create_table(table)  # Make an API request.
        print("Created table {}.{}.{}".format(table.project, table.dataset_id, table.table_id))

def initialize_variables():
    """
    Initialize environment variables.
    Returns:
        tuple: The project id and bucket name.
    """
    project_id = os.getenv("PROJECT_ID")
    bucket_name = os.getenv("BUCKET_NAME")
    return project_id, bucket_name

def initialize_client_and_bucket(bucket_name):
    """
    Initialize a storage client and get a bucket object.
    Args:
        bucket_name (str): The name of the bucket.
    Returns:
        tuple: The storage client and bucket object.
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    return storage_client, bucket

def load_tokeniser(bucket, pickle_file_path='model_store/tokenizerV1.pkl'):
    """
    Load normalization stats from a blob in the bucket.
    Args:
        bucket (Bucket): The bucket object.
        pickle_file (str): The name of the blob containing the stats.
    Returns:
        dict: The loaded stats.
    """
    print("Pulling Tokenizer from storage")
    local_temp_file = "temp_tokenizer.pkl"

    # Download the pickle file from the bucket
    blob = bucket.blob(pickle_file_path)
    blob.download_to_filename(local_temp_file)

    # Load the pickle file
    with open(local_temp_file, 'rb') as file:
        tokenizer = pickle.load(file)

    # Clean up: Remove the local temporary file after loading
    os.remove(local_temp_file)
    print("Downloaded Tokeniser")
    return tokenizer

def load_model(bucket, bucket_name, models_prefix='model/gcp_'):
    """
    Fetch and load the latest model from the bucket.
    Args:
        bucket (Bucket): The bucket object.
        bucket_name (str): The name of the bucket.
    Returns:
        _BaseEstimator: The loaded model.
    """

    # List all blobs in the models directory and find the latest model
    blobs = list(bucket.list_blobs(prefix=models_prefix))
    model_folders = {}
    for blob in blobs:
        match = re.search(r'gcp_model_(\d+)-(\d+)', blob.name)
        if match:
            print("Match Name: ", blob.name)
            timestamp = int(match.group(1) + match.group(2))  # Concatenate the timestamp
            model_folders[timestamp] = blob.name.split('/')[1]  # Extract folder name

    if not model_folders:
        raise Exception("No model folders found in the specified bucket and prefix.")

    latest_model_folder = model_folders[max(model_folders.keys())]
    print("Latest Model: ", latest_model_folder)
    
    model_dir = f'gs://finer_data_bk/model/{latest_model_folder}/'
    print("Model Directory: ", model_dir)
    model = tf.keras.models.load_model(model_dir)
    print("Loaded Model")
    # Clean up: Remove the local directory after loading the model
    # shutil.rmtree(local_model_dir)

    return model

def process_string(s):
    # Check if the string represents a number (integer or decimal)
    pattern = r'^\d{1,3}(,\d{3})*(\.\d+)?$|^(\d+\.?\d*)$'
    # Check if the string represents a number
    if re.match(pattern, s):
        return "[num]"
    else:
        return s.lower()
    
def process_list(input_list):
    return [process_string(s) for s in input_list]

@app.route(os.environ['AIP_HEALTH_ROUTE'], methods=['GET'])
def health_check():
    """Health check endpoint that returns the status of the server.
    Returns:
        Response: A Flask response with status 200 and "healthy" as the body.
    """
    return {"status": "healthy"}

@app.route(os.environ['AIP_PREDICT_ROUTE'], methods=['POST'])
def predict():
    """
    Prediction route that normalizes input data, and returns model predictions.
    Returns:
        Response: A Flask response containing JSON-formatted predictions.
    """
    prediction_start_time = time.time()
    request_json = request.get_json()
    request_instances = request_json['instances']
    text_inp = request_instances[0]['text_input']
    input = text_inp.split()
    logger.log_text(f'Input Data: {input}', severity='INFO')
    np_len = len(input)
    pro_inp = process_list(input)
    tok_inp = tokeniser.texts_to_sequences([pro_inp])
    print(tok_inp[0])
    oov_words = tok_inp[0].count(tokeniser.word_index['<OOV>'])
    pad_inp = pad_sequences(tok_inp, maxlen=32, padding='post')
    prediction = model.predict(pad_inp)
    label_prediction = np.argmax(prediction, axis = 2)[0][:np_len].tolist()
    current_timestamp = datetime.now().isoformat()
    prediction_end_time = time.time()
    prediction_latency = prediction_end_time - prediction_start_time
    prediction = {"predictions":label_prediction}
    logger.log_text(f"Prediction results: {prediction}", severity='INFO')
    rows_to_insert = [{
        "Text" : text_inp,
        "length": np_len,
        "timestamp": current_timestamp,
        "oov_words": oov_words,
        "warning_flag":0,
        "latency":prediction_latency,
        "predictions": label_prediction
    }]
    print(rows_to_insert)
    errors = bq_client.insert_rows_json(table_id, rows_to_insert)
    if errors == []:
        logger.log_text("New predictions inserted into BigQuery.", severity='INFO')
    else:
        logger.log_text(f"Encountered errors inserting predictions into BigQuery: {errors}", severity='ERROR')
    return jsonify(prediction)
    

project_id, bucket_name = initialize_variables()
print(project_id, bucket_name)
storage_client, bucket = initialize_client_and_bucket(bucket_name)
tokeniser = load_tokeniser(bucket)
model = load_model(bucket, bucket_name)
schema = get_table_schema()
create_table_if_not_exists(bq_client, table_id, schema)

if __name__ == '__main__':
    print("Started predict.py ")
    app.run(host='0.0.0.0', port=8080)
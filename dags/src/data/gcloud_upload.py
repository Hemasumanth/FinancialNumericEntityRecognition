from google.cloud import storage
from pathlib import Path
import os
def upload_to_gcs(bucket_name, source_file, destination_blob_name, logger):
    """Uploads a file to Google Cloud Storage if it doesn't already exist.

    Args:
        bucket_name (str): Name of the bucket.
        source_file (Path): Local path to the file.
        destination_blob_name (str): Destination path name in the bucket.
        logger (logging.Logger): Logger object for logging messages.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    try:
        if not blob.exists():
            blob.upload_from_filename(str(source_file))
            logger.info(f"File {source_file} uploaded to {destination_blob_name} in bucket {bucket_name}.")
        else:
            logger.info(f"File {destination_blob_name} already exists in bucket {bucket_name}. Skipping upload.")
    except Exception as e:
        logger.error(f"Error uploading file {source_file}: {e}")

def upload_files(PROJECT_FOLDER, BUCKET_NAME, logger):
    file_mappings = {
        PROJECT_FOLDER / 'data' / 'final' / 'train_token.json': 'train/train_token.json',
        PROJECT_FOLDER / 'data' / 'final' / 'test_token.json': 'test/test_token.json',
        PROJECT_FOLDER / 'model_store' / 'tokenizerV1.pkl': 'model_store/tokenizerV1.pkl',
        PROJECT_FOLDER / 'model_store' / 'stats.json' : 'model_store/stats.json'
    }

    for src, dest in file_mappings.items():
        upload_to_gcs(BUCKET_NAME, src, dest, logger)
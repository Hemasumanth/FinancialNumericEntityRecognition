import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.beam.impl import Context
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.tf_metadata import dataset_metadata, schema_utils
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys
import ast

root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))
from src.data import DATA_FOLDER, DESTINATION_FOLDER

FINAL_FOLDER = DATA_FOLDER / 'final'
FINAL_FOLDER.mkdir(parents=True, exist_ok=True)

def merge_data():
    try:
        train_csv = pd.read_csv(DESTINATION_FOLDER / 'train.csv')
        test_csv = pd.read_csv(DESTINATION_FOLDER / 'test.csv')
        valid_csv = pd.read_csv(DESTINATION_FOLDER / 'validation.csv')
        df = pd.concat([train_csv, test_csv, valid_csv], ignore_index=True)
    except Exception as e:
        print(f"Issues in loading raw data files. Error: {e}")
        return None
    
    final_df = df.drop_duplicates(subset=['tokens'])
    print(final_df.head())
    return final_df

def preprocessing_fn(inputs):
    """Preprocess input columns into transformed columns."""
    tokens = inputs['tokens']
    ner_tags = inputs['ner_tags']

    # Convert tokens from string representation of a list to an actual list of strings
    tokens = tf.strings.split(tf.strings.strip(tokens), delimiter=',')
    ner_tags = tf.strings.split(tf.strings.strip(ner_tags), delimiter=',')

    # Regex pattern for matching integers or floats
    number_pattern = r'^[-+]?\d*\.\d+|\d+$'

    # Convert tokens: if they match the number pattern, replace with [num], else convert to lowercase
    tokens = tf.where(
        tf.strings.regex_full_match(tokens, number_pattern),
        '[num]',
        tf.strings.lower(tokens)
    )

    return {
        'tokens': tokens,
        'ner_tags': ner_tags
    }

def main():
    df = merge_data()
    if df is None:
        print("Error in merging data.")
        return

    # Convert your dataframe to a tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices(dict(df))
    
    # Define metadata
    feature_spec = {
        'tokens': tf.io.FixedLenFeature([], tf.string),
        'ner_tags': tf.io.FixedLenFeature([], tf.string)
    }
    metadata = dataset_metadata.DatasetMetadata(
        schema_utils.schema_from_feature_spec(feature_spec)
    )
    
    # Apply the transformations
    transformed_dataset, transform_fn = (
        (dataset, metadata) | beam_impl.AnalyzeAndTransformDataset(preprocessing_fn)
    )
    transformed_data, transformed_metadata = transformed_dataset

    # Convert the transformed data back to a DataFrame for further processing or model training
    transformed_df = pd.DataFrame(list(transformed_data))

    # Split the data
    train_df, test_df = train_test_split(transformed_df, test_size=0.2)

    # ... continue with any further operations on train_df and test_df ...
    train_df.to_csv(FINAL_FOLDER / 'train_transformed.csv', index=False)
    test_df.to_csv(FINAL_FOLDER / 'test_transformed.csv', index=False)

    print(f"Saved transformed train data to {FINAL_FOLDER / 'train_transformed.csv'}")
    print(f"Saved transformed test data to {FINAL_FOLDER / 'test_transformed.csv'}")

if __name__ == "__main__":
    main()

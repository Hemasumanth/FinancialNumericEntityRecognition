
import pandas as pd
from sklearn.model_selection import train_test_split

def merge_data(PROJECT_FOLDER):
    
    """
    Merges and de-duplicates data from raw CSV files.

    This function reads three CSV files (train, test, and validation) located in the 'data/raw' directory 
    specified by PROJECT_FOLDER. It then concatenates these files and removes duplicate rows based on the 'tokens' column.

    :param PROJECT_FOLDER: A Path object representing the project's root directory.
    :return: A DataFrame containing the merged and de-duplicated data.
    """
    
    try:
        RAW_DATA_FOLDER = PROJECT_FOLDER / 'data' / 'raw'
        train_csv = pd.read_csv(RAW_DATA_FOLDER / 'train.csv')
        test_csv = pd.read_csv(RAW_DATA_FOLDER / 'test.csv')
        valid_csv = pd.read_csv(RAW_DATA_FOLDER / 'validation.csv')
        df = pd.concat([train_csv, test_csv, valid_csv], ignore_index=True)
    except Exception as e:
        print(f"Issues in loading raw data files. Error: {e}")
        return None
    
    final_df = df.drop_duplicates(subset=['tokens']) # Remove duplicate rows based on the 'tokens' column
    return final_df

def split_data(PROJECT_FOLDER, logger):
    
    """
    Splits the merged data into training and testing sets and saves them as JSON files.

    This function checks if the training and testing JSON files already exist in the 'data/inter' directory 
    specified by PROJECT_FOLDER. If they do not exist, it calls the merge_data function to merge the raw data 
    and then splits it into training and testing sets. The resulting sets are saved as JSON files in the 
    'data/inter' directory.

    :param PROJECT_FOLDER: A Path object representing the project's root directory.
    """
    
    INT_FOLDER = PROJECT_FOLDER / 'data' / 'inter'
    train_path = INT_FOLDER / 'train.json'
    test_path = INT_FOLDER / 'test.json'
    INT_FOLDER.mkdir(parents=True, exist_ok=True) # Create the 'data/inter' directory if it doesn't exist
    
    # Check if the files exist
    if not train_path.exists() and not test_path.exists():
        logger.info('Started splitting data into train and test data.')
        df = merge_data(PROJECT_FOLDER) # Calls merge_data function to get the merged data
        if df is not None:
            train_df, test_df = train_test_split(df, test_size=0.2, random_state = 42)  # Splits data into train and test sets
            test_df.to_json(test_path, orient='records', lines=True)  # Save the test set as a JSON file
            train_df.to_json(train_path, orient='records', lines=True)  # Save the train set as a JSON file
            logger.info("Train and test data split and saved successfully.")
        else:
            print("Something went wrong with the merge_data function.")
            logger.warning("Something went wrong with the merge_data function.")
    else:
        logger.info("Train and test JSON files already exist. Skipping the processing.")
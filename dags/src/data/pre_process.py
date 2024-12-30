import pandas as pd
import ast
import re
from collections import Counter
import pickle

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

def conver_to_list(PROJECT_FOLDER, FILE, logger):
    
    """
    Convert string data in the 'tokens' column to lists of processed tokens and save as a JSON file.

    This function reads data from a JSON file in the 'data/inter' directory, converts the 'tokens' column from a
    string representation to a list, processes the tokens, and saves the data as a JSON file in the 'data/final' directory.

    :param PROJECT_FOLDER: A Path object representing the project's root directory.
    :param FILE: A string representing the name of the JSON file to be processed.
    """
    
    try:
        INTER_FOLDER = PROJECT_FOLDER / 'data' / 'inter'

        FINAL_FOLDER = PROJECT_FOLDER / 'data' / 'final'
        FINAL_FOLDER.mkdir(parents=True, exist_ok=True)
        
        PATH_FILE = FINAL_FOLDER/f'{FILE}_pre_process.json'
        if(PATH_FILE.exists()):
            print("Data already exists")
            logger.info("Data already exists")
            return None

        df = pd.read_json(INTER_FOLDER / f'{FILE}.json', lines = True)
        if(df is None):
            logger.error(f"Empty Data from {FILE}.json")
            raise Exception("Data is empty")
        
        logger.info('Started converting columns data from string to list.')
        df['tokens'] = df['tokens'].apply(ast.literal_eval)
        logger.info('Completed tokens columns data from string to list.')

        logger.info('Starting tokens processing.')
        df['tokens'] = df['tokens'].apply(process_list)
        print('Completed tokens processing.')

        logger.info('Started converting ner_tags data from string to list..')
        df['ner_tags'] = df['ner_tags'].apply(ast.literal_eval)
        print('Completed ner_tags columns data from string to list.')
        
        df = df[df['ner_tags'].apply(lambda x: sum(x) != 0)]
        
        df.to_json(PATH_FILE, orient='records', lines=True)
        logger.info(f"Data saved to {PATH_FILE}")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except pd.errors.EmptyDataError:
        logger.warning("Input JSON file is empty.")
    except Exception as e:
        logger.error(f"Issues in converting columns data from string to list. Error: {e}")
        return None
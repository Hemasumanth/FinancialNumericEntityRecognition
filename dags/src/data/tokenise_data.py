import pickle
import pandas as pd
def load_tokniser(TOKENIZER_FILE):
    if not TOKENIZER_FILE.exists():
        return None
    with open(TOKENIZER_FILE, "rb") as file:
        loaded_tokenizer = pickle.load(file)
    return loaded_tokenizer

def tokenise_json_data(PROJECT_FOLDER, logger):
    TOKENIZER_FILE = PROJECT_FOLDER / 'model_store' / 'tokenizerV1.pkl'
    logger.info("Loading tokenizer")
    tokeniser = load_tokniser(TOKENIZER_FILE)
    if tokeniser is None:
        logger.info("Tokenizer not found")
        return None
    # Assuming PROJECT_FOLDER and other variables are already defined
    logger.info("Tokenising Train Data")
    train_data_path = PROJECT_FOLDER / 'data' / 'final' / 'train_pre_process.json'
    train_tokenized_path = PROJECT_FOLDER / 'data' / 'final' / 'train_token.json'

    # Check if the tokenized file already exists
    if not train_tokenized_path.exists():
        df = pd.read_json(train_data_path, lines=True)
        df['token_data'] = tokeniser.texts_to_sequences(df['tokens'])
        df.to_json(train_tokenized_path, orient='records', lines=True)
        logger.info('Saved tokenised Train data to: {}'.format(train_tokenized_path))
    else:
        logger.info('Tokenized Train data already exists at: {}'.format(train_tokenized_path))

    logger.info("Tokenising Test Data")
    test_data_path = PROJECT_FOLDER / 'data' / 'final' / 'test_pre_process.json'
    test_tokenized_path = PROJECT_FOLDER / 'data' / 'final' / 'test_token.json'

    # Check if the tokenized file already exists
    if not test_tokenized_path.exists():
        df = pd.read_json(test_data_path, lines=True)
        df['token_data'] = tokeniser.texts_to_sequences(df['tokens'])
        df.to_json(test_tokenized_path, orient='records', lines=True)
        logger.info('Saved tokenised Test data to: {}'.format(test_tokenized_path))
    else:
        logger.info('Tokenized Test data already exists at: {}'.format(test_tokenized_path))

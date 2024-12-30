import pandas as pd
import numpy as np
from src.data.tokenise_data import tokenise_json_data
from datetime import datetime

def data_stats(PROJECT_FOLDER, LOGGER):
    FINAL_FOLDER = PROJECT_FOLDER / 'data' / 'final'
    FILE = 'train_pre_process.json'
    df = pd.read_json(FINAL_FOLDER / f'train_pre_process.json', lines = True)
    
    word_list = df['tokens']
    # Remove punctuations from list
    sentences = list()
    for i in word_list:
        sentence = ' '.join(i)
        sentences.append(sentence)
    sent_lns = [len(sentence.split()) for sentence in sentences]
    # Get min length of sentences
    min_len = np.min(sent_lns)
    # Get max length of sentences
    max_len = np.max(sent_lns)
    # Get average length of sentences
    avg_len = np.mean(sent_lns)
    # Get std of length of sentences
    std_len = np.std(sent_lns)
    # Get 95th percentile of sentence lengths
    len_95_perc = int(np.percentile(sent_lns, 95))
    tokenise_json_data(PROJECT_FOLDER, LOGGER)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    df = pd.read_json(FINAL_FOLDER / f'train_token.json', lines = True)
    
    stats_dict = {'min_len':min_len,'max_len':max_len,'std_len':std_len, 'avg_len':avg_len, '95th_perc': len_95_perc, 'timestamp':datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
    stats = pd.DataFrame([stats_dict])
    FILE_PATH = PROJECT_FOLDER / 'model_store'  / f'stats.json'
    if FILE_PATH.exists():    
        stats_old = pd.read_json(FILE_PATH, lines = True)
        stats = pd.concat([stats_old, stats])
    stats.to_json(FILE_PATH, orient='records', lines=True)


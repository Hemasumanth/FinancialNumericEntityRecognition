import pandas as pd
from pathlib import Path
import sys
root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))
from src.data.download_data import download_and_store_data

if __name__ == "__main__":
    download_and_store_data('nlpaueb/finer-139')
import pytest
from dags.src.data.data_split import merge_data, split_data
from unittest.mock import patch, Mock
from pathlib import Path
import pandas as pd

# Fixtures to mock CSV data
@pytest.fixture
def mock_csv_data():
    data = {
        'tokens': ['token1', 'token1', 'token2', 'token3'],  
        'other_column': [1, 1, 2, 3]
    }
    return pd.DataFrame(data)



# Test for merge_data function
def test_merge_data(tmpdir, mock_csv_data):
    # Mocking pandas read_csv function
    with patch('pandas.read_csv', return_value=mock_csv_data):
        project_folder = Path(tmpdir)
        result_df = merge_data(project_folder)

        # Check if duplicates are removed
        assert len(result_df) == 3
        assert 'token1' in result_df['tokens'].values
        assert 'token2' in result_df['tokens'].values
        assert 'token3' in result_df['tokens'].values

# Test for split_data function
def test_split_data(tmpdir, mock_csv_data):
        rootdir = Path(tmpdir)
        mock_logger = Mock()
        #@patch('merge_data')
        with patch('dags.src.data.data_split.merge_data', return_value=mock_csv_data):   
            split_data(rootdir, mock_logger)
            #assert (rootdir / 'data' / 'inter').exists()
            # Check if train and test path exist
            assert (rootdir / 'data' / 'inter' / 'train.json').exists()
            assert (rootdir / 'data' / 'inter' / 'test.json').exists()


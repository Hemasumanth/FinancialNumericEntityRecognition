import pytest
from dags.src.data.download import download_and_store_data
from unittest.mock import Mock, patch
from pathlib import Path
import pandas as pd
# Test to ensure function downloads and stores data correctly
def test_download_and_store_success(tmpdir):
    # Mocking the logger and datasets library
    mock_logger = Mock()
    mock_load = Mock(return_value=pd.DataFrame({'tokens': ['["token1", "token2"]', '["token3", "token4"]']}))

    # Using pytest's tmpdir fixture for a temporary directory
    root_dir = Path(tmpdir)

    with patch('datasets.load_dataset', mock_load):
        download_and_store_data("dummy_dataset", mock_logger, root_dir)
        # Check if files are created
        assert (root_dir / 'data' / 'raw' / 'train.csv').exists()
        assert (root_dir / 'data' / 'raw' / 'validation.csv').exists()
        assert (root_dir / 'data' / 'raw' / 'test.csv').exists()
        # Check logger calls
        mock_logger.info.assert_called()
        mock_logger.error.assert_not_called()

# Test to verify handling of exceptions
def test_download_and_store_exception(tmpdir):
    # Mocking the logger and datasets library to raise an exception
    mock_logger = Mock()
    mock_datasets = Mock()
    mock_datasets.load_dataset.side_effect = Exception("Download error")

    root_dir = Path(tmpdir)

    with patch('datasets.load_dataset', mock_datasets), pytest.raises(Exception):
        download_and_store_data("dummy_dataset", mock_logger, root_dir)
        mock_logger.error.assert_called_with("An error occurred: Download error")


# Test to ensure function skips existing files
def test_download_and_store_skip_existing(tmpdir):
    # Mocking the logger and datasets library
    mock_logger = Mock()
    mock_datasets = Mock()

    root_dir = Path(tmpdir)

    # Pre-create one of the files
    (root_dir / 'data' / 'raw').mkdir(parents=True)
    with open(root_dir / 'data' / 'raw' / 'train.csv', 'w') as f:
        f.write("dummy data")
    with open(root_dir / 'data' / 'raw' / 'test.csv', 'w') as f:
        f.write("dummy data")
    with open(root_dir / 'data' / 'raw' / 'validation.csv', 'w') as f:
        f.write("dummy data")

    with patch('datasets.load_dataset', mock_datasets):
        download_and_store_data("dummy_dataset", mock_logger, root_dir)
        # Ensure the existing file was skipped
        mock_logger.info.assert_any_call("train.csv already exists at {}. Skipping download.".format(root_dir / 'data' / 'raw' / 'train.csv'))
        # Check that other files are created
        assert (root_dir / 'data' / 'raw' / 'validation.csv').exists()
        assert (root_dir / 'data' / 'raw' / 'test.csv').exists()
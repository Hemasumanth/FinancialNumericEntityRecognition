import pytest
from dags.src.data.pre_process import conver_to_list, process_list, process_string
from unittest.mock import patch, Mock
from pathlib import Path
import pandas as pd
import json

# Test for conver_to_list function to skip if the file already exists
def test_conver_to_list_skip_existing(tmpdir):
    # Mocking pandas read_csv function and to_csv method
    mock_read_json = Mock(return_value= ({'tokens': ['["token1", "token2"]', '["token3", "token4"]']}))
    mock_to_json = Mock()
    mock_logger = Mock()

    with patch('pandas.read_json', mock_read_json), \
         patch('pandas.DataFrame.to_json', mock_to_json):
        project_folder = Path(tmpdir)
        file_name = "test"
        # Pre-create one of the files
        (project_folder / 'data' / 'final').mkdir(parents=True)
        with open(project_folder / 'data' / 'final' / 'test_pre_process.json', 'w') as f:
            f.write("dummy data") 
        result = conver_to_list(project_folder, file_name, mock_logger)

        # Check if the function returned None (indicating success)
        assert result is None


# Test for process_list function
def test_process_list():
    input_list = ["123", "Token"]
    expected_output = ["[num]", "token"]
    assert process_list(input_list) == expected_output

# Parametrized test for the process_string function
@pytest.mark.parametrize("test_input, expected", [
    ("123", "[num]"),
    ("123.456", "[num]"),
    ("ABC", "abc"),
    ("Hello World!", "hello world!")
])
def test_process_string(test_input, expected):
    assert process_string(test_input) == expected

# Setup the files and paths used in pre_process function
# This setup creates and inter and final folder, 
# within a temp directory acting as the project/root folder
# It also creates a sample dataset and places it in the inter directory,
# as the fuction tries to read data from the same
# We return the project folder
@pytest.fixture
def setup_files(tmp_path):
    # Create a temporary directory structure and files for testing
    project_folder = tmp_path / "project"
    inter_folder = project_folder / "data" / "inter"
    inter_folder.mkdir(parents=True)
    final_folder = project_folder / "data" / "final"
    final_folder.mkdir(parents=True)
    # Create a sample JSON file for testing
    sample_data = [{'tokens': '["token1", "token2"]',
                    'ner_tags': '[0, 1]'}]
    sample_file_path = inter_folder / "test.json"
    with open(sample_file_path, 'w') as file:
        for item in sample_data:
            json.dump(item, file)
            file.write('\n')
    return project_folder

# Test if the function calls necessary functions and gives no errors
def test_conver_to_list_json_token(setup_files):
    project_folder = setup_files
    mock_logger = Mock()
    mock_to_json = Mock()
    #mock_read_json = Mock(return_value = sample_data)
    with patch('pandas.DataFrame.to_json', mock_to_json):
        file_name = "test"
        result = conver_to_list(project_folder, file_name, mock_logger)
        # Check if to_json is called to write processed data
        assert mock_to_json.called
        mock_logger.info.assert_called()
        mock_logger.error.assert_not_called()

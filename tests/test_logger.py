import pytest
from dags.src.data.logger_info import setup_logger
from unittest.mock import patch, Mock
from pathlib import Path
import logging

# Test for setup_logger function
def test_setup_logger(tmpdir):
    # Mocking Path to use tmpdir for PROJECT_ROOT
    mock_project_root = Path(tmpdir)
    logger_name = "test_logger"

    with patch('pathlib.Path', return_value=mock_project_root):
        logger = setup_logger(mock_project_root, logger_name)

        # Check if the logger is correctly configured
        assert logger.name == logger_name
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) == 2  # Console and File handlers

        # Check if the file handler is set up correctly
        file_handler = [h for h in logger.handlers if isinstance(h, logging.FileHandler) or isinstance(h, logging.handlers.RotatingFileHandler)]
        assert len(file_handler) == 1
        assert file_handler[0].level == logging.DEBUG
        assert (mock_project_root / 'logs/app.log').exists()
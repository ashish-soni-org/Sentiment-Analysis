import logging
from logging.handlers import RotatingFileHandler
import os
from from_root import from_root
from datetime import datetime

LOG_DIR_NAME = "logs"
LOG_FILE_NAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" # string format time
MAX_Log_FILE_SIZE = 5 * 1024 * 1024
BACKUP_FILE_COUNT = 3 

log_dir_path = os.path.join(from_root(), LOG_DIR_NAME)
os.makedirs(log_dir_path, exist_ok= True)
log_file_path = os.path.join(log_dir_path, LOG_FILE_NAME)

def configure_logger():
    """
    Configure application-wide logging with both file and console handlers.

    This function sets up a root logger with the following configuration:
    - Logging level: DEBUG (captures all log levels)
    - File logging using RotatingFileHandler with automatic log rotation based on size
    - Console logging for real-time output during execution
    - Unified log message format including timestamp, logger name, level, and message

    The rotating file handler writes logs to `log_file_path`, creating new files when
    the size exceeds `MAX_Log_FILE_SIZE` and retaining up to `BACKUP_FILE_COUNT` backups.
    """

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

    file_handler = RotatingFileHandler(log_file_path, maxBytes= MAX_Log_FILE_SIZE, backupCount= BACKUP_FILE_COUNT)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

configure_logger()
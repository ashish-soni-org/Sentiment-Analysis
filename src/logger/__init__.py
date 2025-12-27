import logging
import os
from logging.handlers import RotatingFileHandler
from from_root import from_root
from datetime import datetime

LOG_DIR_NAME = "logs"
LOG_FILE_NAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
MAX_LOG_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_FILE_COUNT = 3 

log_dir_path = os.path.join(from_root(), LOG_DIR_NAME)
os.makedirs(log_dir_path, exist_ok=True)
log_file_path = os.path.join(log_dir_path, LOG_FILE_NAME)

def configure_logger():
    """
    Configure application-wide logging with both file and console handlers.
    """
    logger = logging.getLogger()
    
    if logger.hasHandlers():
        return

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

    # File Handler
    file_handler = RotatingFileHandler(
        log_file_path, 
        maxBytes=MAX_LOG_FILE_SIZE, 
        backupCount=BACKUP_FILE_COUNT
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO) # Change to INFO to reduce console noise

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Silence noisy libraries
    logging.getLogger("watchfiles").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

configure_logger()
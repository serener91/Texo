import os
import logging
from logging.handlers import RotatingFileHandler


def get_logger(name: str, log_file: str) -> logging.Logger:

    """
    Each logger writes only to its own file
    Handlers aren't duplicated on reload
    Logs go to different files depending on module
    """

    if not os.path.exists(log_file):
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    # Defines a namespace for logs
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s")

        # file_handler = logging.FileHandler(log_file)

        # Automatically manages log size and rotation
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,  # Rotate log file when it exceeds 5 MB
            backupCount=3              # Keep 3 old files, discard older
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger

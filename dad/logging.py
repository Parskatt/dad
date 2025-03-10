import logging
import sys
from logging.handlers import RotatingFileHandler

# First, create logger for your package
logger = logging.getLogger("DaD")
logger.propagate = False  # Prevent propagation to avoid double logging
logger.addHandler(logging.NullHandler())  # Default null handler


def configure_logger(
    level=logging.INFO,
    log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    date_format="%Y-%m-%d %H:%M:%S",
    file_path=None,
    file_max_bytes=10485760,  # 10MB
    file_backup_count=3,
    stream=sys.stderr,
    propagate=False,  # Default to False to prevent double logging
):
    """
    Configure the package logger with handlers similar to basicConfig.
    This does NOT use basicConfig() and only affects this package's logger.
    """
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        if not isinstance(handler, logging.NullHandler):
            logger.removeHandler(handler)

    # Set propagation
    logger.propagate = propagate

    # Set level
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(log_format, date_format)

    # Add console handler if stream is specified
    if stream:
        console_handler = logging.StreamHandler(stream)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if file path is specified
    if file_path:
        file_handler = RotatingFileHandler(
            file_path, maxBytes=file_max_bytes, backupCount=file_backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

import logging
import os
import sys
from datetime import datetime
from colorama import Fore, Style, init


class CustomFormatter(logging.Formatter):
    """ Custom log formatter with colors for console output. """

    FORMAT = "{asctime} [{threadName:12}] [{levelname:8}] {message}"
    DATEFMT = "%Y-%m-%d %H:%M:%S"

    FORMATS = {
        logging.DEBUG: Fore.CYAN + FORMAT + Style.RESET_ALL,
        logging.INFO: Fore.BLUE + FORMAT + Style.RESET_ALL,
        logging.WARNING: Fore.YELLOW + FORMAT + Style.RESET_ALL,
        logging.ERROR: Fore.RED + FORMAT + Style.RESET_ALL,
        logging.CRITICAL: Fore.RED + Style.BRIGHT + FORMAT + Style.RESET_ALL
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMAT)
        formatter = logging.Formatter(log_fmt, datefmt=self.DATEFMT, style="{")
        return formatter.format(record)

# misc.py
def setup_logger(log_dir="logs"):
    """Set up a logger with both console and file handlers."""
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Generate a unique log filename
    log_filename = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")

    # Create a named logger (isolated from root)
    logger = logging.getLogger("main_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # File handler (non-colored)
    file_handler = logging.FileHandler(log_filename, mode='a')
    file_formatter = logging.Formatter(
        "{asctime} [{threadName:12}] [{levelname:8}] {message}",
        datefmt="%Y-%m-%d %H:%M:%S",
        style="{"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler (colored)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(CustomFormatter())
    logger.addHandler(console_handler)

    return logger
# Example usage
#log = setup_logger(log_dir="my_logs")  # Change folder as needed
#log.info("Logger initialized successfully.")
#log.warning("This is a warning message.")
#log.error("This is an error message.")

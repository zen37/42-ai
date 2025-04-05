import logging
import os

def initialize_log_file(log_file_path, header):
    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(log_file_path):
        with open(log_file_path, "w") as f:
            f.write(header + "\n")

def configure_logging(config):
    # Initialize log files
    log_failure_path = config["log"]["failure"]["path"]
    log_summary_path = config["log"]["summary"]["path"]
    log_format = config["log"]["format"]
    date_format = config["log"]["date_format"]

    initialize_log_file(log_failure_path, config["log"]["failure"]["header"])
    initialize_log_file(log_summary_path, config["log"]["summary"]["header"])

    # Configure loggers
    logging.basicConfig(level=logging.INFO)
    
    # Create loggers for failure and summary
    failure_logger = logging.getLogger("failure_logger")
    summary_logger = logging.getLogger("summary_logger")

    # Set the logging level for each logger
    failure_logger.setLevel(logging.INFO)
    summary_logger.setLevel(logging.INFO)

    # Set up handlers for each log file
    failure_handler = logging.FileHandler(log_failure_path, mode="a")
    summary_handler = logging.FileHandler(log_summary_path, mode="a")

    # Configure formatters using the format and date format from the config
    formatter = logging.Formatter(log_format, datefmt=date_format)
    failure_handler.setFormatter(formatter)
    summary_handler.setFormatter(formatter)
    
    # Add handlers to loggers
    failure_logger.addHandler(failure_handler)
    summary_logger.addHandler(summary_handler)

    return failure_logger, summary_logger
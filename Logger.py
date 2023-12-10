import logging

class Logger:
    def __init__(self, log_file='./log/training_log.log'):
        # Create a custom logger
        self.logger = logging.getLogger(__name__)

        # Set the level of the logger
        self.logger.setLevel(logging.INFO)

        # Create a file handler for writing logs to a file
        file_handler = logging.FileHandler(log_file)

        # Set a format for the log messages
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        self.logger.addHandler(file_handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

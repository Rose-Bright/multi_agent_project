import logging
import sys

def setup_logging(level=logging.INFO, log_file=None):
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    for handler in handlers:
        handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    for handler in handlers:
        root_logger.addHandler(handler)

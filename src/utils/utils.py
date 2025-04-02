# utils.py
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO, filename='../../results/logs/training.log')
    return logging.getLogger(__name__)

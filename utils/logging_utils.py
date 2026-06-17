import logging
import os

def setup_logging(log_name='pipeline', log_dir='output'):
    """
    Sets up logging to both a file in the specified log_dir and the console.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{log_name}.log')
    
    # Remove existing handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(log_name)

# Utility functions for dealing with printing to screen 
import sys 
from contextlib import contextmanager, nullcontext, redirect_stdout
import contextlib
import logging 
import datetime 
from pathlib import Path 
import io

# Log if logger available, else print
def log_print(logger: logging.Logger, 
             level: str = 'info',
             message: str = '', 
             *args, **kwargs) -> None:
    if logger:
        log_method = getattr(logger, level, None)
        if log_method:
            log_method(message, *args, **kwargs)
    elif level in ['error', 'warning']:
        print(message)
            
def get_logger(log_name: str, 
               log_dir: Path = None,
               level: str = 'info' 
            ) -> logging.Logger: 
    today = datetime.date.today()
    today_str = today.strftime('%Y-%m-%d')

    # Get logger object 
    logger = logging.getLogger(log_name)
    
    # Set logging level 
    level_obj = logging.getLevelNamesMapping()[level.upper()]
    logger.setLevel(level_obj)
    
    # Configure logger 
    if log_dir is not None: 
        handler = logging.FileHandler(f'{log_name}_{today_str}.log')
    else: 
        handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    return logger

def print_metadata(args, title: str = 'OPERATION'):
    """Print metadata based on argparse inputs"""
    print("=" * 50)
    print(f"{title} METADATA")
    print("=" * 50)
    for key, value in vars(args).items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("=" * 50)
    print()

@contextmanager
def suppress_stdout(enable = True):
    if enable: 
        with redirect_stdout(io.StringIO()):
            yield
    else: 
        with nullcontext():
            yield
    # OLD: Deprecate 
    # with open(os.devnull, "w") as devnull:
    #     old_stdout = sys.stdout
    #     sys.stdout = devnull
    #     try:
    #         yield
    #     finally:
    #         sys.stdout = old_stdout
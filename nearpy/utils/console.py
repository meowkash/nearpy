# Utility functions for dealing with printing to screen 
import os 
import sys 
import contextlib

def print_metadata(args):
    """Print metadata based on argparse inputs"""
    print("=" * 50)
    print("OPERATION METADATA")
    print("=" * 50)
    for key, value in vars(args).items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("=" * 50)
    print()

@contextlib.contextmanager
def suppress_stdout():
    """Context manager to suppress stdout"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
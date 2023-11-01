import sys
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))  # Replace __file__ with your script's path if not in the same directory

# Get the directory above the current directory
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)
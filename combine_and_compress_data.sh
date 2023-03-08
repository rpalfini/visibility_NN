#!/bin/bash

# Set the variable with the command-line argument
ARGUMENT="$1"

# Run the Python command with the argument
python3 csv_file_combiner.py "$ARGUMENT" 

# Set the output filename using string concatenation
OUTPUT_FILENAME="${ARGUMENT}_merge.csv"

# Compress the output file using 7z with desired filename
7z a "$OUTPUT_FILENAME".7z "$OUTPUT_FILENAME"

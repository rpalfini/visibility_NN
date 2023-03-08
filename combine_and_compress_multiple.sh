#!/bin/bash

# Array of input filenames
# INPUT_FILES=(file1.csv file2.csv file3.csv)

# Accept input filenames from the command line
INPUT_FILES=("$@")


# Loop over input filenames and run the script in parallel
for FILENAME in "${INPUT_FILES[@]}"
do
  # Run the Python command with the argument
  python script.py "$FILENAME"

  # Set the output filename using string concatenation
  OUTPUT_FILENAME="${FILENAME%.*}_merge.csv"

  # Compress the output file using 7z with desired filename
  7z a "$OUTPUT_FILENAME.7z" "$OUTPUT_FILENAME" &
done

# Wait for all parallel processes to complete
wait

# Notify user that all files have been zipped
echo "All files have been zipped."

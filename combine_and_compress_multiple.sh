#!/bin/bash

# Accept input directory names from the command line
INPUT_DIRS=("$@")

# Loop over input directory names and run the script in parallel
for DIRNAME in "${INPUT_DIRS[@]}"
do
  # Run the Python command with the directory name as argument
  python3 csv_file_combiner.py "$DIRNAME"

  # Set the output filename to the input directory name

  FOLDER_NAME=$(basename "$DIRNAME")
  OUTPUT_FILENAME="${FOLDER_NAME}_merge.csv"

  ZIP_NAME=$(basename "$DIRNAME" _merge.csv)

  # Compress the output directory using 7z with desired filename
  7z a "./results_merge/$ZIP_NAME.7z" "./results_merge/$OUTPUT_FILENAME"
done

# Wait for all parallel processes to complete
wait

# Notify user that all directories have been zipped
echo "All directories have been zipped."

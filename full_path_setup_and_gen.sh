#!/bin/bash

today=$(date '+%y_%m_%d')
base_path="/home/robert_palfini/environment/visibility_NN"
directory="$base_path/data_out"
file_name_prefix=${today}_aws_batch
batch_num=1

# check if the folder exists in the directory
while [ -d "${directory}/${file_name_prefix}${batch_num}" ]; do
  # increment the iteration variable and update the folder name
  ((batch_num++))
done
echo "date: $today"
echo "batch_num: $batch_num"
file_name="${file_name_prefix}${batch_num}"

python3 $base_path/obstacle_course_gen.py -f "$file_name" -nc 5 -no 20 -o -u -b

# Set the prefix to search for
dir="$base_path/obs_courses"
prefix=$file_name

# Count the number of files that match the prefix
num_files=$(ls -1 "${dir}/${prefix}_"* 2>/dev/null | wc -l)

# create log file directory
log_dir="$base_path/run_logs"

if [ ! -d "$log_dir" ]; then
    mkdir "$log_dir"
    echo "Directory created: $log_dir"
else
    echo "Directory already exists: $log_dir"
fi

# Run a for loop for each file that matches the prefix
for (( ii=0; ii<num_files; ii++ ))
do
    test_file=$base_path/${file_name}_${ii}.txt
    log_file=$base_path/${log_dir}/${file_name}.log
    error_file=$base_path/${log_dir}/${file_name}_error.log
    python3 vis_main.py "$test_file" -b True -gs -f "$file_name" >> "$log_file" 2>> "$error_file" &
    echo "test_file = $test_file"
    echo "file_name = $file_name"
    echo "ii = $ii"
done

echo "num obs files found: $num_files"

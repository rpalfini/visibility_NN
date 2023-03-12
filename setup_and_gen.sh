#!/bin/bash

start_time=$(date +%s)

today=$(date '+%y_%m_%d')
directory="./data_out"
file_name_prefix=${today}_google_batch
batch_num=1

# check if the folder exists in the directory
while [ -d "${directory}/${file_name_prefix}${batch_num}" ]; do
  # increment the iteration variable and update the folder name
  ((batch_num++))
done
echo "date: $today"
echo "batch_num: $batch_num"
file_name="${file_name_prefix}${batch_num}"

python3 obstacle_course_gen.py -f "$file_name" -nc 5 -no 20 -o -u -b

# Set the prefix to search for
dir=./obs_courses
prefix=$file_name

# Count the number of files that match the prefix
num_files=$(ls -1 "${dir}/${prefix}_"* 2>/dev/null | wc -l)

# create log file directory
log_dir="./run_logs"

if [ ! -d "$log_dir" ]; then
    mkdir "$log_dir"
    echo "Directory created: $log_dir"
else
    echo "Directory already exists: $log_dir"
fi

# Run a for loop for each file that matches the prefix
for (( ii=0; ii<num_files; ii++ ))
do
    test_file=${file_name}_${ii}.txt
    log_file=${log_dir}/${file_name}.log
    error_file=${log_dir}/${file_name}_error.log
    # if [ $ii -ge $((num_files-2)) ]; then
    # # Run the Python command with different arguments
    #     python3 vis_main.py "$test_file" -b True -gs -f "$file_name" -a >> "$log_file" 2>> "$error_file" &
    # else
    # Run the Python command with default arguments
    python3 vis_main.py "$test_file" -b True -gs -f "$file_name" >> "$log_file" 2>> "$error_file" &
    # fi
    echo "test_file = $test_file"
    echo "file_name = $file_name"
    echo "ii = $ii"
done

echo "num obs files found: $num_files"
wait
echo "all files proccessed"
end_time=$(date +%s)
duration=$(echo "scale=2; $end_time - $start_time" | bc)
seconds=$(( duration % 60 ))
minutes=$(( (duration / 60) % 60 ))
hours=$(( duration / 3600 ))

echo "Execution time for $file_name: $duration seconds or $hours hours, $minutes minutes, $seconds seconds" | mail -s "$file_name Execution Time" robert.palfini@gmail.com

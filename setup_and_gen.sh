#!/bin/bash

today=$(date '+%y_%m_%d')
directory="./data_out"
file_name_prefix=${today}_aws_batch
batch_num=1

# check if the folder exists in the directory
while [ -d "${directory}/${file_name_prefix}${batch_num}" ]; do
  # increment the iteration variable and update the folder name
  ((batch_num++))
done
echo "$batch_num"
file_name="${file_name_prefix}${batch_num}"

python3 obstacle_course_gen.py -f $file_name -nc 5 -no 20

for ii in {0..19}
do
    test_file=${file_name}_${ii}.txt
    log_file=./run_logs/${file_name}.log
    error_file=./run_logs/${file_name}_error.log
    python3 vis_main.py $test_file -b True -gs -f $file_name >> $log_file 2>> $error_file &
done
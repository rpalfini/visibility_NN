#!/bin/bash
batch_num=6
today=$(date '+%y_%m_%d')
file_name=${today}_aws_batch${batch_num}

for ii in {0..19}
do
    test_file=${file_name}_${ii}.txt
    log_file=${file_name}.log
    error_file=${file_name}_error.log
    python3 vis_main.py test_file -b True -gs -f $file_name > $log_file > $error_file &
done
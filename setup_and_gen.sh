#!/bin/bash

pip install numpy matplotlib pandas bidict
python3 obstacle_course_gen.py -f 23_02_22_aws_batch4 -nc 5 -no 20

for ii in {0..19}
do
    python3 vis_main.py 23_02_22_aws_batch4_$ii.txt -b True -gs &
done
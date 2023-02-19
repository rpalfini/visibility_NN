#!/bin/bash
# cd visibility_NN/
for ii in {0..19}
do
    python vis_main.py 23_02_19_aws_batch2_$ii.txt -b True -gs &
done
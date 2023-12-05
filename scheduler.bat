:: This script is used to run Neural network models one after another
@REM echo Running the first Python program...
@REM python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses20.npy" -b 64 -e 10 -l 0.0001 -n 20 -o 0 -m 17 -s -sf -sv 30 2> error_first.txt
@REM echo First program finished.

@REM echo Running the second Python program...
@REM python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses1.npy" -b 64 -e 10 -l 0.0001 -n 1 -o 0 -m 16 -s -sf -sv 30 2> error_second.txt
@REM echo Second program finished.

@REM echo Running the third Python program...
@REM python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses2.npy" -b 64 -e 10 -l 0.0001 -n 2 -o 0 -m 17 -s -sf -sv 30 2> error_third.txt
@REM echo Third program finished.

@REM echo Running the fourth Python program...
@REM python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses3.npy" -b 64 -e 10 -l 0.0001 -n 3 -o 0 -m 4 -s -sf -sv 30 2> error_fourth.txt
@REM echo Fourth program finished.

@REM echo Running the fifth Python program...
@REM python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses5.npy" -b 64 -e 10 -l 0.0001 -n 5 -o 0 -m 1 -s -sf -sv 30 2> error_fifth.txt
@REM echo Fifth program finished.

@REM echo Running the sixth Python program...
@REM python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses10.npy" -b 64 -e 10 -l 0.0001 -n 10 -o 0 -m 9 -s -sf -sv 30 2> error_sixth.txt
@REM echo Sixth program finished.

@REM echo Running the seventh Python program...
@REM python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses15.npy" -b 64 -e 10 -l 0.0001 -n 15 -o 0 -m 8 -s -sf -sv 30 2> error_seventh.txt
@REM echo Seventh program finished.

@REM echo Running the eighth Python program...
@REM python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses20.npy" -b 64 -e 10 -l 0.0001 -n 20 -o 0 -m 2 -tv -s -sf -sv 30 2> error_eighth.txt
@REM echo Eighth program finished.

@REM echo Running the ninth Python program...
@REM python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses20.npy" -b 64 -e 10 -l 0.0001 -n 20 -o 0 -m 18 -tv -s -sf -sv 30 2> error_ninth.txt
@REM echo Ninth program finished.

@REM echo Running the tenth Python program...
@REM python ./ML_code/keras_twenty_obst_NN.py -f "E:/main_folder/Augmented Data Sets/double_data_and_shift_inputs/train/double_augmented_train_main_data_file_courses20.npy" -b 256 -e 70 -l 0.0001 -n 20 -o 0 -m 3 -tv -s -sf -sv 30 -tf "E:\main_folder\Augmented Data Sets\double_data_and_shift_inputs\test\main_data_file_courses20.npy" -vf "E:\main_folder\Augmented Data Sets\double_data_and_shift_inputs\validation\validation_main_data_file_courses20.npy" 2> error_eleventh.txt 2> error_tenth.txt
@REM echo Tenth program finished.

@REM echo Running the eleventh Python program...
@REM python ./ML_code/keras_twenty_obst_NN.py -f "E:/main_folder/Augmented Data Sets/double_data_and_shift_inputs/train/double_augmented_train_main_data_file_courses20.npy" -b 256 -e 70 -l 0.0001 -n 20 -o 0 -m 21 -tv -s -sf -sv 30 -tf "E:\main_folder\Augmented Data Sets\double_data_and_shift_inputs\test\main_data_file_courses20.npy" -vf "E:\main_folder\Augmented Data Sets\double_data_and_shift_inputs\validation\validation_main_data_file_courses20.npy" 2> error_eleventh.txt
@REM echo Eleventh program finished.

@REM echo Running the twelth Python program...
@REM python ./ML_code/keras_twenty_obst_NN.py -f "E:/main_folder/Augmented Data Sets/double_data_and_shift_inputs/train/double_augmented_train_main_data_file_courses20.npy" -b 256 -e 70 -l 0.0001 -n 20 -o 0 -m 20 -tv -s -sf -sv 30 -tf "E:\main_folder\Augmented Data Sets\double_data_and_shift_inputs\test\main_data_file_courses20.npy" -vf "E:\main_folder\Augmented Data Sets\double_data_and_shift_inputs\validation\validation_main_data_file_courses20.npy" 2> error_twelth.txt
@REM echo Twelth program finished.

@REM echo Running the thirteenth Python program...
@REM ::python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses3.csv" -b 64 -e 20 -n 3 -m 2 -l 0.001 2> error_third.txt
@REM ::python ./ML_code/keras_twenty_obst_NN.py -f "./ML_code/Data/small_main_data_file_courses3.csv" -b 64 -e 20 -n 3 -m 2 -l 0.001 2> error_third.txt
@REM python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses20.csv" -b 64 -e 20 -l 0.0001 -n 20 -o 0 -m 11 -s 2> error_thirteenth.txt
@REM echo Thirteenth program finished.

echo All programs have been executed.
@REM :: This script is used to run Neural network models one after another
@REM echo Running the first Python program...
@REM ::python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses3.csv" -b 64 -e 15 -n 3 -m 1 -l 0.0001 2> error_first.txt
@REM ::python ./ML_code/keras_twenty_obst_NN.py -f "./ML_code/Data/small_main_data_file_courses3.csv" -b 64 -e 15 -n 3 -m 1 -l 0.0001 2> error_first.txt
@REM python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses1.csv" -b 64 -e 10 -l 0.0001 -n 1 -o 0 -m 16 -s 2> error_first.txt
@REM echo First program finished.

@REM echo Running the second Python program...
@REM ::python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses3.csv" -b 64 -e 15 -n 3 -m 2 -l 0.01 2> error_second.txt
@REM ::python ./ML_code/keras_twenty_obst_NN.py -f "./ML_code/Data/small_main_data_file_courses3.csv" -b 64 -e 10 -n 3 -m 2 -l 0.01 2> error_second.txt
@REM python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses2.csv" -b 64 -e 10 -l 0.0001 -n 2 -o 0 -m 17 -s 2> error_second.txt
@REM echo Second program finished.

@REM echo Running the third Python program...
@REM ::python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses3.csv" -b 64 -e 20 -n 3 -m 2 -l 0.001 2> error_third.txt
@REM ::python ./ML_code/keras_twenty_obst_NN.py -f "./ML_code/Data/small_main_data_file_courses3.csv" -b 64 -e 20 -n 3 -m 2 -l 0.001 2> error_third.txt
@REM python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses3.csv" -b 64 -e 10 -l 0.0001 -n 3 -o 0 -m 4 -sf -sv 30 2> error_third.txt
@REM echo Third program finished.

@REM echo Running the fourth Python program...
@REM ::python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses3.csv" -b 64 -e 20 -n 3 -m 2 -l 0.001 2> error_third.txt
@REM ::python ./ML_code/keras_twenty_obst_NN.py -f "./ML_code/Data/small_main_data_file_courses3.csv" -b 64 -e 20 -n 3 -m 2 -l 0.001 2> error_third.txt
@REM python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses4.csv" -b 64 -e 10 -l 0.0001 -n 4 -o 0 -m 1 -sf -sv 30  2> error_fourth.txt
@REM echo Fourth program finished.

@REM echo Running the fifth Python program...
@REM ::python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses3.csv" -b 64 -e 20 -n 3 -m 2 -l 0.001 2> error_third.txt
@REM ::python ./ML_code/keras_twenty_obst_NN.py -f "./ML_code/Data/small_main_data_file_courses3.csv" -b 64 -e 20 -n 3 -m 2 -l 0.001 2> error_third.txt
@REM python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses5.csv" -b 64 -e 10 -l 0.0001 -n 5 -o 0 -m 1 -s 2> error_fifth.txt
@REM echo Fifth program finished.

@REM echo Running the sixth Python program...
@REM ::python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses3.csv" -b 64 -e 20 -n 3 -m 2 -l 0.001 2> error_third.txt
@REM ::python ./ML_code/keras_twenty_obst_NN.py -f "./ML_code/Data/small_main_data_file_courses3.csv" -b 64 -e 20 -n 3 -m 2 -l 0.001 2> error_third.txt
@REM python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses10.csv" -b 64 -e 10 -l 0.0001 -n 10 -o 0 -m 9 -s 2> error_sixth.txt
@REM echo Sixth program finished.

@REM echo Running the seventh Python program...
@REM ::python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses3.csv" -b 64 -e 20 -n 3 -m 2 -l 0.001 2> error_third.txt
@REM ::python ./ML_code/keras_twenty_obst_NN.py -f "./ML_code/Data/small_main_data_file_courses3.csv" -b 64 -e 20 -n 3 -m 2 -l 0.001 2> error_third.txt
@REM python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses10.csv" -b 64 -e 10 -l 0.0001 -n 10 -o 0 -m 7 -s 2> error_seventh.txt
@REM echo Seventh program finished.

@REM echo Running the eighth Python program...
@REM ::python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses3.csv" -b 64 -e 20 -n 3 -m 2 -l 0.001 2> error_third.txt
@REM ::python ./ML_code/keras_twenty_obst_NN.py -f "./ML_code/Data/small_main_data_file_courses3.csv" -b 64 -e 20 -n 3 -m 2 -l 0.001 2> error_third.txt
@REM python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses20.csv" -b 64 -e 10 -l 0.0001 -n 20 -o 0 -m 8 -s -sf -sv 30 2> error_eighth.txt

@REM echo Eighth program finished.

@REM echo Running the ninth Python program...
@REM ::python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses3.csv" -b 64 -e 20 -n 3 -m 2 -l 0.001 2> error_third.txt
@REM ::python ./ML_code/keras_twenty_obst_NN.py -f "./ML_code/Data/small_main_data_file_courses3.csv" -b 64 -e 20 -n 3 -m 2 -l 0.001 2> error_third.txt
@REM python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses20.csv" -b 64 -e 10 -l 0.0001 -n 20 -o 0 -m 8 -sf -sv 30 2> error_ninth.txt
@REM echo Ninth program finished.

@REM echo Running the tenth Python program...
@REM ::python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses3.csv" -b 64 -e 20 -n 3 -m 2 -l 0.001 2> error_third.txt
@REM ::python ./ML_code/keras_twenty_obst_NN.py -f "./ML_code/Data/small_main_data_file_courses3.csv" -b 64 -e 20 -n 3 -m 2 -l 0.001 2> error_third.txt
@REM python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses20.csv" -b 64 -e 10 -l 0.001 -n 20 -o 2 -m 8 -s 2> error_tenth.txt
@REM echo Tenth program finished.

@REM echo Running the eleventh Python program...
@REM ::python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses3.csv" -b 64 -e 20 -n 3 -m 2 -l 0.001 2> error_third.txt
@REM ::python ./ML_code/keras_twenty_obst_NN.py -f "./ML_code/Data/small_main_data_file_courses3.csv" -b 64 -e 20 -n 3 -m 2 -l 0.001 2> error_third.txt
@REM python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses20.csv" -b 64 -e 10 -l 0.0001 -n 20 -o 0 -m 18 -s 2> error_eleventh.txt
@REM echo Eleventh program finished.

echo Running the twelth Python program...
::python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses3.csv" -b 64 -e 20 -n 3 -m 2 -l 0.001 2> error_third.txt
::python ./ML_code/keras_twenty_obst_NN.py -f "./ML_code/Data/small_main_data_file_courses3.csv" -b 64 -e 20 -n 3 -m 2 -l 0.001 2> error_third.txt
python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses20.csv" -b 64 -e 200 -l 0.0001 -n 20 -o 0 -m 8 -s 2> error_twelth.txt
echo Twelth program finished.

echo Running the thirteenth Python program...
::python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses3.csv" -b 64 -e 20 -n 3 -m 2 -l 0.001 2> error_third.txt
::python ./ML_code/keras_twenty_obst_NN.py -f "./ML_code/Data/small_main_data_file_courses3.csv" -b 64 -e 20 -n 3 -m 2 -l 0.001 2> error_third.txt
python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses20.csv" -b 64 -e 200 -l 0.0001 -n 20 -o 0 -m 8 -s -sp 0.8 0.1 0.1 2> error_thirteenth.txt
echo Thirteenth program finished.

echo All programs have been executed.
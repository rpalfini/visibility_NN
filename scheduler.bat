:: This script is used to run Neural network models one after another
echo Running the first Python program...
::python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses3.csv" -b 64 -e 15 -n 3 -m 1 -l 0.0001 2> error_first.txt
::python ./ML_code/keras_twenty_obst_NN.py -f "./ML_code/Data/small_main_data_file_courses3.csv" -b 64 -e 15 -n 3 -m 1 -l 0.0001 2> error_first.txt
python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses20.csv" -b 64 -e 20 -l 0.0001 -n 20 -o 0 -m 0 2> error_first.txt
echo First program finished.

echo Running the second Python program...
::python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses3.csv" -b 64 -e 15 -n 3 -m 2 -l 0.01 2> error_second.txt
::python ./ML_code/keras_twenty_obst_NN.py -f "./ML_code/Data/small_main_data_file_courses3.csv" -b 64 -e 10 -n 3 -m 2 -l 0.01 2> error_second.txt
python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses20.csv" -b 64 -e 30 -l 0.0001 -n 20 -o 0 -m 1 2> error_second.txt
echo Second program finished.

echo Running the third Python program...
::python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses3.csv" -b 64 -e 20 -n 3 -m 2 -l 0.001 2> error_third.txt
::python ./ML_code/keras_twenty_obst_NN.py -f "./ML_code/Data/small_main_data_file_courses3.csv" -b 64 -e 20 -n 3 -m 2 -l 0.001 2> error_third.txt
python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses20.csv" -b 64 -e 30 -l 0.0001 -n 20 -o 0 -m 2 2> error_third.txt
echo Third program finished.

echo Running the fourth Python program...
::python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses3.csv" -b 64 -e 20 -n 3 -m 2 -l 0.001 2> error_third.txt
::python ./ML_code/keras_twenty_obst_NN.py -f "./ML_code/Data/small_main_data_file_courses3.csv" -b 64 -e 20 -n 3 -m 2 -l 0.001 2> error_third.txt
python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses20.csv" -b 64 -e 30 -l 0.0001 -n 20 -o 0 -m 3 2> error_fourth.txt
echo Fourth program finished.

echo Running the fifth Python program...
::python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses3.csv" -b 64 -e 20 -n 3 -m 2 -l 0.001 2> error_third.txt
::python ./ML_code/keras_twenty_obst_NN.py -f "./ML_code/Data/small_main_data_file_courses3.csv" -b 64 -e 20 -n 3 -m 2 -l 0.001 2> error_third.txt
python ./ML_code/keras_twenty_obst_NN.py -f "D:/Vis_network_data/data_file_by_course/main_data_file_courses20.csv" -b 64 -e 50 -l 0.0001 -n 20 -o 0 -m 4 2> error_fifth.txt
echo Fifth program finished.

echo All programs have been executed.
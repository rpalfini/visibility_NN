@REM :: This script is used to run Neural network models one after another
echo Running the first Python program...
@REM python ./ML_code/keras_twenty_obst_NN.py -f "E:/main_folder/data_file_by_course/main_data_file_courses1.npy" -b 256 -e 10 -l 0.0001 -n 1 -o 0 -m 16 -s -sf -sv 30 -tv 2> error_first.txt
python ./ML_code/keras_train_generator.py -xd "D:/Vis_network_data/Augmented Data Sets/generator_exp_double_data_and_shift_inputs/train/feat_triple_augmented_train_main_data_file_courses20.npy" -yd "D:/Vis_network_data/Augmented Data Sets/generator_exp_double_data_and_shift_inputs/train/label_triple_augmented_train_main_data_file_courses20.npy" -f "D:/Vis_network_data/Augmented Data Sets/generator_exp_double_data_and_shift_inputs/train/triple_augmented_train_main_data_file_courses20.npy" -b 256 -e 10 -l 0.0001 -n 20 -o 0 -m 21 -tv -s -tf "D:/Vis_network_data/Augmented Data Sets/double_data_and_shift_inputs/test/main_data_file_courses20.npy" -vf "D:/Vis_network_data/Augmented Data Sets/double_data_and_shift_inputs/validation/validation_main_data_file_courses20.npy" 2> error_first.txt
echo First program finished.

echo Running the second Python program...
@REM python ./ML_code/keras_twenty_obst_NN.py -f "E:/main_folder/data_file_by_course/main_data_file_courses2.npy" -b 256 -e 10 -l 0.0001 -n 2 -o 0 -m 17 -s -sf -sv 30 -tv 2> error_second.txt
python ./ML_code/keras_train_generator.py -xd "D:/Vis_network_data/Augmented Data Sets/generator_exp_double_data_and_shift_inputs/train/feat_double_augmented_train_main_data_file_courses20.npy" -yd "D:/Vis_network_data/Augmented Data Sets/generator_exp_double_data_and_shift_inputs/train/label_double_augmented_train_main_data_file_courses20.npy" -f "D:/Vis_network_data/Augmented Data Sets/generator_exp_double_data_and_shift_inputs/train/double_augmented_train_main_data_file_courses20.npy" -b 256 -e 10 -l 0.0001 -n 20 -o 0 -m 21 -tv -s -tf "D:/Vis_network_data/Augmented Data Sets/double_data_and_shift_inputs/test/main_data_file_courses20.npy" -vf "D:/Vis_network_data/Augmented Data Sets/double_data_and_shift_inputs/validation/validation_main_data_file_courses20.npy" 2> error_second.txt
echo Second program finished.

@REM echo Running the third Python program...
@REM python ./ML_code/keras_twenty_obst_NN.py -f "E:/main_folder/data_file_by_course/main_data_file_courses3.npy" -b 256 -e 10 -l 0.0001 -n 3 -o 0 -m 21 -s -sf -sv 30 -tv 2> error_third.txt
@REM echo Third program finished.

@REM echo Running the fourth Python program...
@REM python ./ML_code/keras_twenty_obst_NN.py -f "E:/main_folder/data_file_by_course/main_data_file_courses3.npy" -b 256 -e 10 -l 0.0001 -n 3 -o 0 -m 23 -s -sf -sv 30 -tv 2> error_fourth.txt
@REM echo Fourth program finished.

@REM echo Running the fifth Python program...
@REM python ./ML_code/keras_twenty_obst_NN.py -f "E:/main_folder/data_file_by_course/main_data_file_courses5.npy" -b 256 -e 10 -l 0.0001 -n 5 -o 0 -m 21 -s -sf -sv 30 -tv 2> error_fifth.txt
@REM echo Fifth program finished.

@REM echo Running the sixth Python program...
@REM python ./ML_code/keras_twenty_obst_NN.py -f "E:/main_folder/data_file_by_course/main_data_file_courses10.npy" -b 256 -e 10 -l 0.0001 -n 10 -o 0 -m 21 -s -sf -sv 30 -tv 2> error_sixth.txt
@REM echo Sixth program finished.

@REM echo Running the seventh Python program...
@REM python ./ML_code/keras_twenty_obst_NN.py -f "E:/main_folder/data_file_by_course/main_data_file_courses15.npy" -b 256 -e 10 -l 0.0001 -n 15 -o 0 -m 21 -s -sf -sv 30 -tv 2> error_seventh.txt
@REM echo Seventh program finished.

@REM echo Running the eighth Python program...
@REM python ./ML_code/keras_twenty_obst_NN.py -f "E:/main_folder/data_file_by_course/main_data_file_courses16.npy" -b 256 -e 10 -l 0.0001 -n 16 -o 0 -m 21 -s -sf -sv 30 -tv 2> error_eighth.txt
@REM echo Eighth program finished.

@REM echo Running the ninth Python program...
@REM python ./ML_code/keras_twenty_obst_NN.py -f "E:/main_folder/data_file_by_course/main_data_file_courses17.npy" -b 256 -e 10 -l 0.0001 -n 17 -o 0 -m 21 -s -sf -sv 30 -tv 2> error_ninth.txt
@REM echo Ninth program finished.

echo Running the tenth Python program...
python ./ML_code/keras_twenty_obst_NN.py -f "E:/main_folder/data_file_by_course/main_data_file_courses18.npy" -b 256 -e 10 -l 0.0001 -n 18 -o 0 -m 21 -s -sf -sv 30 -tv 2> error_tenth.txt
echo Tenth program finished.

echo Running the eleventh Python program...
python ./ML_code/keras_twenty_obst_NN.py -f "E:/main_folder/data_file_by_course/main_data_file_courses19.npy" -b 256 -e 10 -l 0.0001 -n 19 -o 0 -m 21 -s -sf -sv 30 -tv 2> error_eleventh.txt
echo Eleventh program finished.

echo Running the twelth Python program...
python ./ML_code/keras_twenty_obst_NN.py -f "E:/main_folder/data_file_by_course/main_data_file_courses20.npy" -b 256 -e 10 -l 0.0001 -n 20 -o 0 -m 21 -sf -sv 30 -tv 2> error_twelth.txt
python ./ML_code/keras_twenty_obst_NN.py -f "E:/main_folder/data_file_by_course/main_data_file_courses20.npy" -b 256 -e 10 -l 0.0001 -n 20 -o 0 -m 21 -s -sf -sv 30 -tv 2> error_twelth_2.txt
echo Twelth program finished.

echo Running the thirteenth Python program...
python ./ML_code/keras_train_generator.py -xd "D:/Vis_network_data/Augmented Data Sets/generator_exp_double_data_and_shift_inputs/train/feat_triple_augmented_train_main_data_file_courses20.npy" -yd "D:/Vis_network_data/Augmented Data Sets/generator_exp_double_data_and_shift_inputs/train/label_triple_augmented_train_main_data_file_courses20.npy" -f "D:/Vis_network_data/Augmented Data Sets/generator_exp_double_data_and_shift_inputs/train/triple_augmented_train_main_data_file_courses20.npy" -b 256 -e 50 -l 0.0001 -n 20 -o 0 -m 21 -tv -s -tf "D:/Vis_network_data/Augmented Data Sets/double_data_and_shift_inputs/test/main_data_file_courses20.npy" -vf "D:/Vis_network_data/Augmented Data Sets/double_data_and_shift_inputs/validation/validation_main_data_file_courses20.npy" 2> error_thirteenth.txt
echo Thirteenth program finished.

@REM python ./ML_code/evaluate_all_n_less_N_values.py -f "./main_train_results/main_data_file_courses20/model_17" -b 64 -n 20

@REM python split_obs_by_num.py

@REM python save_csv_to_numpy.py -f "E:/main_folder/to_be_added_by_course" -s "E:/main_folder/npy_to_be_added_by_course"

@REM python ./ML_code/evaluate_keras_model.py -n 20 -d "E:/main_folder/npy_to_be_added_by_course/main_data_file_courses20.npy" -f "./main_train_results/main_data_file_courses20/model_17" -b 64 -s

@REM python ./ML_code/evaluate_all_n_less_N_values.py -f "./main_train_results/main_data_file_courses20/model_17" -b 64 -n 20 -mr 1 3
@REM python ./ML_code/evaluate_all_n_less_N_values.py -f "./main_train_results/main_data_file_courses20/model_17" -b 64 -n 20 -mr 4 6
@REM python ./ML_code/evaluate_all_n_less_N_values.py -f "./main_train_results/main_data_file_courses20/model_17" -b 64 -n 20 -mr 7 9
@REM python ./ML_code/evaluate_all_n_less_N_values.py -f "./main_train_results/main_data_file_courses20/model_17" -b 64 -n 20 -mr 10 12
@REM python ./ML_code/evaluate_all_n_less_N_values.py -f "./main_train_results/main_data_file_courses20/model_17" -b 64 -n 20 -mr 13 15
@REM python ./ML_code/evaluate_all_n_less_N_values.py -f "./main_train_results/main_data_file_courses20/model_17" -b 64 -n 20 -mr 16 18
@REM python ./ML_code/evaluate_all_n_less_N_values.py -f "./main_train_results/main_data_file_courses20/model_17" -b 64 -n 20 -mr 19 20

echo All programs have been executed.
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np

import config # needed to load modules from other local packages
import csv_file_combiner as cfc
# from ML_code import util
import util
import save_csv_to_numpy as scn

'''finds index where courses change as well as outputs those rows to file and tells youthe number of samples per data course'''
def main():

    # Assuming you have your data as a NumPy array
    # Replace the following array with your actual data
    # data = util.load_data("D:/Vis_network_data/data_file_by_course/main_data_file_courses20.npy")
    # data = util.load_data("E:/main_folder/npy_to_be_added_by_course/main_data_file_courses20.npy")
    # data = util.load_data("ML_code/Data/small_main_data_file_courses20.csv")
    
    # file_path = "D:/Vis_network_data/Augmented Data Sets/double_data_and_shift_inputs/train/double_augmented_train_main_data_file_courses20.npy"
    file_path = "D:/Vis_network_data/Augmented Data Sets/double_data_and_shift_inputs/train/triple_augmented_train_main_data_file_courses20.npy"
    # file_path = "E:/main_folder/Augmented Data Sets/double_data_and_shift_inputs/main_data_file_courses20.npy"
    # file_path = "E:/main_folder/Augmented Data Sets/double_data_and_shift_inputs/train/train_main_data_file_courses20.npy"
    
    
    data = util.load_data(file_path)
    


    # Define the columns you're interested in
    columns_of_interest = np.arange(4,64)

    # Use np.diff to find changes between consecutive rows in the specified columns
    changes = np.diff(data[:, columns_of_interest], axis=0)

    # Find rows where changes occur by checking if any element in the resulting array is non-zero
    rows_with_changes = np.any(changes != 0, axis=1)

    # Get the rows where values change
    result = data[1:][rows_with_changes]

    # Get the row indices where values change
    result_indices = np.where(rows_with_changes)[0] + 1  # Adding 1 to account for zero-based indexing

    index_differences = np.diff(result_indices)

    num_points = 0
    for i in range(len(index_differences) - 1, -1, -1):
        # print(index_differences[i])
        if num_points >= 792265:
            print(f'last index was {i}.')
            print(f'num_points needed {num_points}')
            print(f'total courses - last index = {index_differences.shape[0] - i}')
            break
        
        num_points += index_differences[i]

    # Print or use the result array as needed
    print(result)
    print(result_indices)
    print(index_differences)

    output_file_path = "change_indices_20_obs_file"
    with open(output_file_path, 'w') as file:
        for index in result_indices:
            file.write(f"{index}\n")

    np.savetxt(os.path.join(os.path.dirname(file_path),'20_obs_row_changed_triple.txt'),result,delimiter=',')
    print(f"result.shape = {result.shape}")


if __name__ == "__main__":
    main()
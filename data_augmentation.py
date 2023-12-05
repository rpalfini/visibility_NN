import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm
from ML_code import util


'''This script can augment a dataset by shuffling the obstacle data inputs around.'''


def main(verify_result):
    file_extend_path = "D:/Vis_network_data/Augmented Data Sets/double_data_and_shift_inputs/train/double_augmented_train_main_data_file_courses20.npy"
    file_path = "D:/Vis_network_data/Augmented Data Sets/double_data_and_shift_inputs/train/train_main_data_file_courses20.npy"
    # file_extend_path = "ML_code/Data/double_augmented_small_main_data_file_courses20.csv.npy"
    # file_path = "ML_code/Data/small_main_data_file_courses20.csv"
    num_obs = util.read_model_num_from_file_path(file_path)
    train_data = util.load_data(file_path)

    X, Y = util.separate_features_labels(train_data,num_obs)
    opt_cost = train_data[:,-1]

    X_column_names = []
    X_column_names.append('x_start')
    X_column_names.append('y_start')
    X_column_names.append('x_end')
    X_column_names.append('y_end')

    Y_column_names = []

    for ii in range(1,num_obs+1):
        X_column_names.append(f'x_{ii}')
        X_column_names.append(f'y_{ii}')
        X_column_names.append(f'r_{ii}')
        Y_column_names.append(f'{ii}')

    print(f'number of X column names = {len(X_column_names)}')
    print(f'number of Y column names = {len(Y_column_names)}')

    X_df = pd.DataFrame(X,columns = X_column_names)
    Y_df = pd.DataFrame(Y,columns = Y_column_names)
    print(X_df.head())
    print(Y_df.head())

    # reverse_column_order(verify_result, file_path, num_obs, train_data, opt_cost, X_df, Y_df)
    half_shift_columns(verify_result, file_path, file_extend_path, num_obs, opt_cost, X_df, Y_df)

def half_shift_columns(verify_result, file_path, file_extend_path, num_obs, opt_cost, X_df, Y_df):
    if verify_result:
        #used to verify operation performed
        X_df_original = X_df.copy()
        Y_df_original = Y_df.copy()

    if num_obs%2 == 0:
        num_iter = num_obs//2
    else:
        num_iter = (num_obs-1)//2

    for ii in tqdm(range(0,num_iter),desc="Processing", unit="item"):
        X_df[[f"x_{ii+1}",f"x_{num_iter+ii+1}"]] = X_df[[f"x_{num_iter+ii+1}",f"x_{ii+1}"]]
        X_df[[f"y_{ii+1}",f"y_{num_iter+ii+1}"]] = X_df[[f"y_{num_iter+ii+1}",f"y_{ii+1}"]]
        X_df[[f"r_{ii+1}",f"r_{num_iter+ii+1}"]] = X_df[[f"r_{num_iter+ii+1}",f"r_{ii+1}"]]
        Y_df[[f"{ii+1}",f"{num_iter+ii+1}"]] = Y_df[[f"{num_iter+ii+1}",f"{ii+1}"]]

    if verify_result:
        # verify all columns swapped
        X_truth_array = []
        Y_truth_array = []
        r_truth_array = []
        label_truth_array = []

        for ii in tqdm(range(0,num_iter),desc="Testing", unit="item"):
            if all(X_df_original[f"x_{ii+1}"] == X_df[f"x_{num_iter+ii+1}"]):
                X_truth_array.append(True)
            else:
                X_truth_array.append(False)
            if all(X_df_original[f"y_{ii+1}"] == X_df[f"y_{num_iter+ii+1}"]):
                Y_truth_array.append(True)
            else:
                Y_truth_array.append(False)
            if all(X_df_original[f"r_{ii+1}"] == X_df[f"r_{num_iter+ii+1}"]):
                r_truth_array.append(True)
            else:
                r_truth_array.append(False)
            if all(Y_df_original[f"{ii+1}"] == Y_df[f"{num_iter+ii+1}"]):
                label_truth_array.append(True)
            else:
                label_truth_array.append(False)

        truth_array = [all(X_truth_array), all(Y_truth_array), all(r_truth_array), all(label_truth_array)]
        if all(truth_array):
            print('Operation successful.')
        else:
            print('Operation failed.')
            raise Exception('Operation did not go as planned...')

    X_np = X_df.to_numpy()
    Y_np = Y_df.to_numpy()

    rearranged_data = np.hstack((X_np,Y_np,opt_cost[:,np.newaxis]))

    data2extend = util.load_data(file_extend_path)

    stacked_data = np.vstack((data2extend,rearranged_data))

    np.save(os.path.join(os.path.dirname(file_path),f"triple_augmented_{os.path.basename(file_path)}"),stacked_data)

def reverse_column_order(verify_result, file_path, num_obs, train_data, opt_cost, X_df, Y_df):
    if verify_result:
        #used to verify operation performed
        X_df_original = X_df.copy()
        Y_df_original = Y_df.copy()

    if num_obs%2 == 0:
        num_iter = num_obs//2
    else:
        num_iter = (num_obs-1)//2

    for ii in tqdm(range(0,num_iter),desc="Processing", unit="item"):
        X_df[[f"x_{ii+1}",f"x_{num_obs-ii}"]] = X_df[[f"x_{num_obs-ii}",f"x_{ii+1}"]]
        X_df[[f"y_{ii+1}",f"y_{num_obs-ii}"]] = X_df[[f"y_{num_obs-ii}",f"y_{ii+1}"]]
        X_df[[f"r_{ii+1}",f"r_{num_obs-ii}"]] = X_df[[f"r_{num_obs-ii}",f"r_{ii+1}"]]
        Y_df[[f"{ii+1}",f"{num_obs-ii}"]] = Y_df[[f"{num_obs-ii}",f"{ii+1}"]]

    if verify_result:
        # verify all columns swapped
        X_truth_array = []
        Y_truth_array = []
        r_truth_array = []
        label_truth_array = []

        for ii in tqdm(range(0,num_iter),desc="Processing", unit="item"):
            if all(X_df_original[f"x_{ii+1}"] == X_df[f"x_{num_obs-ii}"]):
                X_truth_array.append(True)
            else:
                X_truth_array.append(False)
            if all(X_df_original[f"y_{ii+1}"] == X_df[f"y_{num_obs-ii}"]):
                Y_truth_array.append(True)
            else:
                Y_truth_array.append(False)
            if all(X_df_original[f"r_{ii+1}"] == X_df[f"r_{num_obs-ii}"]):
                r_truth_array.append(True)
            else:
                r_truth_array.append(False)
            if all(Y_df_original[f"{ii+1}"] == Y_df[f"{num_obs-ii}"]):
                label_truth_array.append(True)
            else:
                label_truth_array.append(False)

        truth_array = [all(X_truth_array), all(Y_truth_array), all(r_truth_array), all(label_truth_array)]
        if all(truth_array):
            print('Operation successful.')
        else:
            print('Operation failed.')
            raise Exception('Operation did not go as planned...')

    X_np = X_df.to_numpy()
    Y_np = Y_df.to_numpy()

    rearranged_data = np.hstack((X_np,Y_np,opt_cost[:,np.newaxis]))

    stacked_data = np.vstack((train_data,rearranged_data))

    np.save(os.path.join(os.path.dirname(file_path),f"double_augmented_{os.path.basename(file_path)}"),stacked_data)
    
if __name__ == "__main__":
    verify_result = True
    main(verify_result)
import os
import numpy as np
from ML_code import util


'''This file splits a data set into two numpy files for train and validation and saves them to train and val folders.'''

def main(file_path,split_percentages):
    dirname = os.path.dirname(file_path)
    basename = os.path.basename(file_path)
    num_obs = util.read_model_num_from_file_path(file_path)
    train_out_path = os.path.join(dirname,'train')
    val_out_path = os.path.join(dirname,"validation")
    
    data = util.load_data(file_path)
    # split_data = util.shuffle_and_split_data(data,num_obs,split_percentages,shuffle_data=False)
    split_data = util.split_array(data,split_percentages)

    # save training file
    np.save(os.path.join(train_out_path,f"train_{basename}"),split_data[0])

    # save validation file
    np.save(os.path.join(val_out_path,f"validation_{basename}"),split_data[1])




if __name__ == "__main__":
    file_path = "E:/main_folder/Augmented Data Sets/double_data_and_shift_inputs/main_data_file_courses20.npy"
    # split_percentages = {"train": 0.9, "val": 0.1, "test": 0}
    # first entry is train percent, second is val percent, and third is test percent. Currently do not save a test percentage because nojt using test data from this set.
    split_percentages = [0.9,0.1,0]
    main(file_path,split_percentages)
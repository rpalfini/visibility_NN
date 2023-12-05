import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm
import util


'''This file loads a numpy file splits it into features and labels, and then saves the features and labels.'''



def main(file_path):
    data = util.load_data(file_path)
    data_type = data.dtype
    shape = data.shape
    print(f'data.shape = {data.shape}')
    basename = os.path.basename(file_path)
    dirname = os.path.dirname(file_path)    
    X,Y = util.separate_features_labels(data,num_obstacles=util.read_model_num_from_file_path(file_path))
    np.save(os.path.join(dirname,f"feat_{basename}"),X)
    np.save(os.path.join(dirname,f"label_{basename}"),Y)
    print(f"X.shape = {X.shape}")
    print(f"Y.shape = {Y.shape}")











if __name__ == "__main__":
    file_path = "D:/Vis_network_data/Augmented Data Sets/generator_exp_double_data_and_shift_inputs/train/triple_augmented_train_main_data_file_courses20.npy"
    main(file_path)
















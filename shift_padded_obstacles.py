import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import csv_file_combiner as cfc
from ML_code import util


def main(args):

    if args.dir_mode:
        if args.file_dir is None:
            raise Exception('To use directory mode, you need to specify a directory to search for csv files with -f option.')
        csv_files = cfc.get_file_list(args.file_dir)
        for fname in csv_files:
            if util.read_model_num_from_file_path(fname) == 20:
                continue
            args.data_path = os.path.join(args.file_dir,fname)
            shift_and_save_data(args)
    else:
        shift_and_save_data(args)

def shift_and_save_data(args):
    new_x_null_center = 15 # this is the new value we are shfiting the x values to see if it helps the classifier.
    data = util.load_data(args.data_path)
    num_obs = util.read_model_num_from_file_path(args.data_path)
    x_idx = util.find_null_x_idx(num_obs,20)
    data[:,x_idx] = new_x_null_center
    file_save_path = os.path.join(args.save_path,f"main_data_file_courses{num_obs}.npy")
    print(f'saving data to {file_save_path}.')
    np.save(file_save_path,data)

def arg_parse():
    parser = ArgumentParser(description="Script used to shift position of null obstacles in csv file to new location in new numpy file.",formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d","--data_path",type=str,default="./ML_code/Data/small_padded_main_data_file_courses10.csv",help="source csv file where you want to shift arguments")
    parser.add_argument("-s","--save_path",type=str,default="./ML_code/Data/shift_null_x_test",help='directory that you want to store your shift data set in')
    parser.add_argument("-b","--dir_mode",action='store_true',help='turns on directory mode where it shifts all csv files in a given directory.  Specify the directory in ')
    parser.add_argument("-f","--file_dir",type=str,default=None,help='Directory to perform operation on all csv files.')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parse()
    main(args)
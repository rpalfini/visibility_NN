import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import csv_file_combiner as cfc
from ML_code import util


def main(args):
    if args.file_dir is None:
        raise Exception('To use directory mode, you need to specify a directory to search for csv files with -f option.')
    csv_files = cfc.get_file_list(args.file_dir)
    for fname in csv_files:
        if fname == "main_data_file_courses20.csv":
            continue
        load_and_save(args.save_path,args.file_dir,fname)

def load_and_save(save_path,file_dir,fname):
    '''Loads file located at file_dir/fname and saves data as .npy file'''
    file_path = os.path.join(file_dir,fname)
    data = util.load_data(file_path)
    num_obs = util.read_model_num_from_file_path(file_path)
    file_save_path = os.path.join(save_path,f"main_data_file_courses{num_obs}.npy")
    print(f'saving data to {file_save_path}.')
    np.save(file_save_path,data)

def arg_parse():
    parser = ArgumentParser(description="Scripts converts all files in directory from csv to numpy and saves in new directory.",formatter_class=ArgumentDefaultsHelpFormatter)
    # parser.add_argument("-d","--data_path",type=str,default="./ML_code/Data/small_padded_main_data_file_courses10.csv",help="source csv file where you want to shift arguments")
    parser.add_argument("-s","--save_path",type=str,default="./ML_code/Data/shift_null_x_test",help='directory that you want to store your shift data set in')
    # parser.add_argument("-b","--dir_mode",action='store_true',help='turns on directory mode where it shifts all csv files in a given directory.  Specify the directory in ')
    parser.add_argument("-f","--file_dir",type=str,default=None,help='Directory to perform operation on all csv files.')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parse()
    main(args)
    
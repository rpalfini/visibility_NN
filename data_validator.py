from visibility_graph import calc_ndata_col
import csv_file_combiner as cfc
import os
from argparse import ArgumentParser

def arg_parse():
    parser = ArgumentParser(description="script checks files in a folder to either verify they are valid or fix based on option selected.  Combines multiple csv data files into one file")
    parser.add_argument("csv_folder_path",help="csv folder to check csv files in")
    parser.add_argument("-f", "--fix", dest='fix_data', action='store_const', const=True, default=False,help="changes validator to fix data files in folder")
    args = parser.parse_args()
    args = vars(args)
    return args

if __name__ == "__main__":
    num_obs = 20
    args = arg_parse()
    csv_folder_path = args["csv_folder_path"]

    csv_file_list = cfc.get_file_list(csv_folder_path)

    is_all_valid = True
    if not args["fix_data"]:
        for csv_file in csv_file_list:
            file_path = os.path.join(csv_folder_path,csv_file)
            is_valid,nrows = cfc.data_check(file_path,calc_ndata_col(20))
            if not is_valid:
                print(f'file {file_path} has {nrows} rows of invalid data')
                is_all_valid = False
    print(f'{csv_folder_path} data is_all_valid = {is_all_valid}')


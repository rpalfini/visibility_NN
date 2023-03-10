import csv_file_combiner as cfc
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def determine_num_obs(line):
    max_num_obs = 20
    tokens = line.split(',')


if __name__ == "__main__":
    fname = "test_data_file.csv"
    fname_no_extension = os.path.splitext(fname)[0]
    # create a dictionary of file handles for the 20 output files
    output_files = {}
    for i in range(1, 21):
        filename = f"{fname_no_extension}_courses{i}.txt"
        output_files[i] = open(filename, "a")

    # read lines from the input file
    file_path = os.path.join("./results_merge",fname)
    data_gen = cfc.csv_reader(file_path)
    for row in data_gen:
        num_obs = determine_num_obs(row)
        output_files[num_obs].write(row)

    # close all output files
    for file_handle in output_files.values():
        file_handle.close()

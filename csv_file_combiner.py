import pandas as pd
import os
import glob

def get_file_list(folder):
    '''Only returns names of csv files in folder'''
    # base_path = 'C:/Users/Robert/Documents/Vis_network_data/'
    # # base_path = 'C:/Users/Robert/git/visibility_NN/results_merge/'
    path = base_path + folder
    extension = 'csv'
    os.chdir(path)
    result = glob.glob('*.{}'.format(extension))
    return result

def get_dir_list(path):
    result = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path,name))]
    return result

def csv_reader(fname):
    for row in open(fname):
        yield row

def data_check(file_path,ncolumns):
    '''checks if every row of file in file_path has ncolumns'''
    data_gen = csv_reader(file_path)
    valid_file = True
    ii = 0
    n_invalid_rows = 0
    for row in data_gen:
        ii += 1
        tokens = row.split(',')
        if len(tokens) != ncolumns:
            valid_file = False
            print(f'row {ii} is invalid with {len(tokens)} tokens')
        for token in tokens:
            if token == '':
                # print(f'row {ii} is invalid')
                n_invalid_rows+=1
                break
    print(f'nrows = {ii}')
    return valid_file,n_invalid_rows

def remove_invalid_data(data_file_path):
    '''input should be absolute or relative path to the data file including the data file i.e. C:/Users/dog.csv'''
    data_file, data_path = split_fname_path(data_file_path)
    data_gen = csv_reader(data_file_path)
    output_file = data_file_path.rstrip(".csv") + "_fixed.csv"

    f = open(output_file,"a")
    n_invalid_rows = 0
    invalid_rows = []
    ii=0
    for row in data_gen:
        is_row_bad = False
        ii+=1
        tokens = row.split(",")
        for token in tokens:
            if token == '':
                n_invalid_rows += 1
                invalid_rows.append(ii)
                is_row_bad = True
                break
        if not is_row_bad:
            f.write(row)
    f.close()
    print(f'outputted corrected data file at {output_file}')
    print(f'{n_invalid_rows} invalid rows at rows: {invalid_rows}')
    is_valid,n_invalid_rows = data_check(output_file,85)
    if not is_valid:
        raise Exception("Fixed file is invalid/has errors")
    else:
        print(f'{data_file_path} is fixed and valid')

def split_fname_path(data_path):
    tokens = data_path.split('/')
    fname = tokens[-1]
    fpath = "/".join(tokens[:-1])
    fpath += "/"
    return fname,fpath

def _name_output(csv_folder):
    if ".csv" in csv_folder:
        output_file = f"{merge_out_folder}{csv_folder[0:21]}_merge.csv"
    else:
        output_file = f"{merge_out_folder}{csv_folder}_merge.csv"
    return output_file

def _chunk_and_output(csv_file_list,output_file,biggest_file=None,):
    
    for csv_file_name in csv_file_list:
        if csv_file_name == biggest_file:
            print('biggest file encountered')
            continue
        else:
            chunk_container = pd.read_csv(csv_file_name, chunksize=CHUNK_SIZE)
            for chunk in chunk_container:
                chunk.to_csv(output_file, mode="a", index=False,header=False)



# def combin
def combine_csv(csv_folder):
    print(csv_folder)

    output_file = _name_output(csv_folder)
    csv_file_list = get_file_list(csv_folder)
    _chunk_and_output(csv_file_list,output_file)


# User options
CHUNK_SIZE = 50000
dir_mode = False
# merge_out_folder = "C:/Users/Robert/git/visibility_NN/results_merge/"
merge_out_folder = "./results_merge"

make_new_file = True # TODO: this mode doesnt work, and is activated by changing to false... this mode creates a new file for outputting.  Use this if all the files are approx the same size
# base_path = 'C:/Users/Robert/Documents/Vis_network_data/'
# base_path = 'C:/Users/Robert/git/visibility_NN/results_merge/'
# base_path = 'C:/Users/Robert/git/visibility_NN/'
base_path = './'
biggest_file = "file1.csv"
biggest_file_path = base_path + biggest_file

if __name__ == "__main__":
    # csv_folder = '23_02_18_19_20'
    csv_folder = '/data_out'
    # csv_folder = '23_02_19_aws_batch1_0_course_1_obs_data.csv[+13]'
    # '23_02_18_batch2'
    # 
    # csv_folder = 'Test'
    # if make_new_file:

    dir_exists = os.path.isdir(merge_out_folder)
    if not dir_exists:
        os.mkdir(merge_out_folder)


    if dir_mode:
        dir_list = get_dir_list('C:/Users/Robert/Documents/Vis_network_data')
        for csv_folder in dir_list:
            print(csv_folder)

            output_file = _name_output(csv_folder)
            csv_file_list = get_file_list(csv_folder,output_file)
            _chunk_and_output(csv_file_list)
    else:
        print(csv_folder)

        output_file = _name_output(csv_folder)
        csv_file_list = get_file_list(csv_folder)
        _chunk_and_output(csv_file_list,output_file)
    # else:
    #     '''This mode appends the other files to the biggest file to save time'''
    #     output_file = biggest_file_path
    #     csv_file_list = get_file_list(csv_folder)
    #     _chunk_and_output(csv_file_list,biggest_file)
        
        


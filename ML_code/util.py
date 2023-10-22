import os
import pickle
import datetime
import sys
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# place to store functions in project
def get_dir_list(path):
    result = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path,name))]
    return result

def init_data_store_folder(data_file):
    data_path = ('./'+data_file)
    dir_exists = os.path.isdir(data_path)
    if dir_exists:
        model_dirs = get_dir_list(data_path)
        model_folder = f'/model_{len(model_dirs)+1}'
        data_store_folder = data_path+model_folder
        os.mkdir(data_store_folder)
    else:
        os.mkdir(data_path)
        model_folder = '/model_1'
        data_store_folder = data_path+model_folder
        os.mkdir(data_store_folder)
    return data_store_folder

def record_model_results(output_dir,epochs, batch_size, learning_rate, train_acc, val_acc, test_acc,
                          model, num_train, num_val, num_test, data_set_name, optimizer_name,start_time):
    with open(output_dir+"/results.txt","w") as f:
        formatted_time = get_datetime()
        f.write(f'{start_time} - {formatted_time}')
        f.write(f'trained on file {data_set_name}\n')
        f.write('train_acc,val_acc,test_acc,epochs,batch_size,optimizer,learning_rate,num_train_data,num_val_data,num_test_data\n')
        f.write(f'{train_acc},{val_acc},{test_acc},{epochs},{batch_size},{optimizer_name},{learning_rate},{num_train},{num_val},{num_test}\n')
        per_train,per_val,per_test = get_data_percents(num_train,num_val,num_test)
        f.write('percent of data for train, val and test\n')
        f.write(f'percent_train={per_train},percent_val={per_val},percent_test={per_test}\n')
        # following code outputs model summary to file
        sys.stdout = f
        model.summary()
        f.write(f'number of data points = {num_train + num_test + num_val}')
    sys.stdout = sys.__stdout__ #reset stdout to console

def record_model_fit_results(results, output_folder):
    model_number,model_results_path = split_fname_path(output_folder)
    fname = f'/{model_number}_results.pkl'
    PK_fname = output_folder + fname
    Temp = open(PK_fname,'wb')
    pickle.dump(results.history,Temp)
    Temp.close()

def get_data_percents(num_train,num_val,num_test):
    total_data = num_train + num_test + num_val
    per_train = num_train/total_data
    per_val = num_val/total_data
    per_test = num_test/total_data
    return per_train, per_val, per_test

def get_datetime(add_new_line=True):
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y/%m/%d %H:%M:%S")
    if add_new_line:
        formatted_datetime += "\n"
    return formatted_datetime

def split_fname_path(data_path):
    '''splits a file name from its path and returns both'''
    tokens = data_path.split('/')
    fname = tokens[-1]
    fpath = "/".join(tokens[:-1])
    fpath += "/"
    return fname,fpath

def arg_parse():
    parser = ArgumentParser(description="Keras Model Training.  Used for script that is training model based on data file",formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", "--num_obs", type=int, default = 3, help="Specify number of obstacles in selected data set")
    parser.add_argument("-f", "--file_path", type=str, default = "./ML_code/Data/main_data_file_courses3.csv")
    parser.add_argument("-b","--batch_size", type=int, default=64, help="set batch size for training")
    parser.add_argument("-e","--n_epochs", type=int, default=100, help="sets number of epochs for the data")
    parser.add_argument("-l","--learning_rate",type=float, default = 0.001, help="sets the learning rate")

    args = parser.parse_args()
    return args

def split_array(original_array, split_percentages):
    if sum(split_percentages) != 1.0:
        raise ValueError("Split percentages must sum to 1.0")

    # Calculate the split indices
    split_indices = [np.round(x * original_array.shape[0]) for x in split_percentages]
    split_indices = np.cumsum(split_indices).astype(int)
    
    # split_indices = np.cumsum(np.round(split_percentages * len(original_array))).astype(int)

    # Perform the array splitting
    splits = np.split(original_array, split_indices)
    splits = splits[0:len(split_percentages)] #remove empty array at the end

    return splits
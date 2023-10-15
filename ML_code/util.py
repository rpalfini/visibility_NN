import os
import pickle
import datetime
import sys
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

def record_model_results(output_dir,epochs, batch_size, train_acc, val_acc, test_acc, model, num_train, num_val, num_test):
    with open(output_dir+"/results.txt","w") as f:
        formatted_time = get_datetime()
        f.write(formatted_time)
        f.write('epochs,batch_size,train_acc,val_acc,num_train_data,num_val_data,num_test_data\n')
        f.write(f'{epochs},{batch_size},{train_acc},{val_acc},{num_train},{num_val},{num_test}\n')
        per_train,per_val,per_test = get_data_percents(num_train,num_val,num_test)
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

def get_datetime():
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y/%m/%d %H:%M:%S")
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
    args = parser.parse_args()
    return args


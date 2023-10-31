import os
import pickle
import datetime
import sys
import numpy as np
from keras.callbacks import ModelCheckpoint
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

## place to store functions used for neural network training

## These functions deal with saving data from training
def arg_parse():
    parser = ArgumentParser(description="Keras Model Training.  Used for script that is training model based on data file",formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", "--num_obs", type=int, default = 3, help="Specify number of obstacles in selected data set")
    parser.add_argument("-f", "--file_path", type=str, default = "./ML_code/Data/small_main_data_file_courses3.csv")
    parser.add_argument("-b","--batch_size", type=int, default=64, help="set batch size for training")
    parser.add_argument("-e","--n_epochs", type=int, default=100, help="sets number of epochs for the data")
    parser.add_argument("-l","--learning_rate",type=float, default = 0.001, help="sets the learning rate")

    args = parser.parse_args()
    return args

def init_data_store_folder(data_file,is_torch=False):
    '''creates directories needed for saving training results'''
    main_results_folder = 'main_train_results'
    if is_torch:
        data_path = os.path.join('.',main_results_folder,'torch_'+data_file)
    else:
        data_path = os.path.join('.',main_results_folder,data_file)
    dir_exists = os.path.isdir(data_path)
    if dir_exists:
        model_dirs = get_dir_list(data_path)
        model_folder = f'model_{len(model_dirs)+1}'
        data_store_folder = os.path.join(data_path,model_folder)
        os.mkdir(data_store_folder)
        make_checkpoint_folder(data_store_folder)
    else:
        # os.mkdir(data_path)
        os.makedirs(data_path,exist_ok=True)
        model_folder = 'model_1'
        data_store_folder = os.path.join(data_path,model_folder)
        os.mkdir(data_store_folder)
        make_checkpoint_folder(data_store_folder)
    return data_store_folder

def make_checkpoint_folder(data_path):
    checkpoint_folder = os.path.join(data_path,'weight_checkpoints')
    os.mkdir(checkpoint_folder)

def get_dir_list(path):
    result = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path,name))]
    return result

def record_model_results(output_dir,epochs, batch_size, learning_rate, train_acc, val_acc, test_acc,
                          model, num_train, num_val, num_test, data_set_name, optimizer_name,start_time):
    # output_path = os.path.join(output_dir,f"results_{df}.txt")
    output_path = make_results_file_name(output_dir,train_acc,val_acc,test_acc)
    with open(output_path,"w") as f:
        formatted_time = get_datetime()
        t_dur = calc_time_duration(start_time,formatted_time)
        f.write(f'{start_time} - {formatted_time}')
        f.write(f'Training Duration = {t_dur}\n')
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

def make_results_file_name(output_dir,train_acc,val_acc,test_acc):
    '''This file creates a results name that includes accuracy of model'''
    train_acc_str = "{:.2f}".format(train_acc)
    val_acc_str = "{:.2f}".format(val_acc)
    test_acc_str = "{:.2f}".format(test_acc)
    fname = os.path.join(output_dir,f"results_{train_acc_str}_{val_acc_str}_{test_acc_str}.txt")
    return fname
    


def record_model_fit_results(results, output_folder):
    # model_number,model_results_path = split_fname_path(output_folder)
    model_results_path, model_number = os.path.split(output_folder)
    fname = f'{model_number}_results.pkl'
    PK_fname = os.path.join(output_folder,fname)
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

def calc_time_duration(start_time, end_time):
    # function expects times formatted by util.get_datetime()
    # removes the \n character
    if start_time.endswith("\n"):
        start_time = start_time.strip()

    if end_time.endswith("\n"):
        end_time = end_time.strip()

    format_str = "%Y/%m/%d %H:%M:%S"
    start = datetime.datetime.strptime(start_time, format_str)
    end = datetime.datetime.strptime(end_time, format_str)

    # Calculate the time duration
    time_duration = end - start
    return time_duration

## These functions are used in preparing datasets
def load_data(file_path):
    return np.loadtxt(file_path,delimiter=',')

def shuffle_and_split_data(dataset,num_obstacles,split_percentages):
    '''Shuffles data using np.random.shuffle and then splits into train, val, and test data.
       INPUTS:
        dataset: file read from load_data
        num_obstacles: number of obstacles in the data file, use size of the padded dataset if using padded data
        split_percentages: length three list indicating the percent for training, validation, and testing
       OUTPUTS:
        split_data: dictionary with train,val,test data as well as opt_costs,num_feat,num_labels'''
    
    np.random.shuffle(dataset)
    # calcualte num labels and features based on data
    features = calc_num_features(num_obstacles)
    labels = num_obstacles

    X = dataset[:,:features]
    Y = dataset[:,features:-1]
    if Y.shape[1] != labels:
        raise Exception(f'incorrect number of labels, expecting {labels} but found {Y.shape[1]}')
    opt_costs = dataset[:,-1] # these are the optimal path costs as found by dijkstra algo during data generation

    X_splits = split_array(X,split_percentages)
    Y_splits = split_array(Y,split_percentages)

    split_data = {}
    split_data["X_train"] = X_splits[0]
    split_data["X_val"] = X_splits[1]
    split_data["X_test"] = X_splits[2]
    
    split_data["Y_train"] = Y_splits[0]
    split_data["Y_val"] = Y_splits[1]
    split_data["Y_test"] = Y_splits[2]

    split_data["opt_costs"] = opt_costs
    split_data["num_features"] = features
    split_data["num_labels"] = labels
    return split_data

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

def calc_num_features(num_obs):
    return 3*num_obs + 4

## General functions
# def split_fname_path(data_path):
#     '''splits a file name from its path and returns both'''
#     tokens = data_path.split('/')
#     fname = tokens[-1]
#     fpath = "/".join(tokens[:-1])
#     fpath += "/"
#     return fname,fpath


    




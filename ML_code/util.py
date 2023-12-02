import sys
import os
import pickle
import datetime
import re
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

## place to store functions used for neural network training

## These functions deal with saving data from training
def arg_parse():
    parser = ArgumentParser(description="NN Model Training.  Used for script that is training model based on data file",formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", "--num_obs", type=int, default = 3, help="Specify number of obstacles in selected data set")
    parser.add_argument("-f", "--file_path", type=str, default = "./ML_code/Data/small_main_data_file_courses3.csv")
    parser.add_argument("-b","--batch_size", type=int, default=64, help="set batch size for training")
    parser.add_argument("-e","--n_epochs", type=int, default=100, help="sets number of epochs for the data")
    parser.add_argument("-l","--learning_rate",type=float, default = 0.001, help="sets the learning rate")
    parser.add_argument("-m", "--NN_model", type=int, default=0, help="Selects neural network to train.  Used for training automation.")
    parser.add_argument("-o","--optimizer", type=int, default=0, help="Selects the optimizer to use for the neural network.  0 is Adam, 1 is RMSprop, 2 is SGD")
    parser.add_argument("-s","--shift", action='store_false', help="Turns off shifting dataset over by half of size of obstacle field.  Expected field size is 30mx30m so shift is -15 to each x,y coordinate")
    parser.add_argument("-xs","--x_shift", action='store_true', help="Turns on shifting dataset over only on x axis by half of size of obstacle field.")
    parser.add_argument("-sf","--scale_flag", action='store_true', help="Turns on scaling inputs based on the argument scale_value.  This scales all of the features by the scale_value.")
    parser.add_argument("-sv","--scale_value", type=float, default=1, help="Scale value used when scale_flag is activated.  it is what the data is divided by. So 30 results in 1/30 scale")
    parser.add_argument("-sp","--split_percentages", type=float, default = [0.90,0.05,0.05], nargs=3, help="enter train/val/test split percentages")

    args = parser.parse_args()
    return args

def fix_path_separator(path_in):
    '''Changes the / to \\ when looking at paths'''
    if os.name == 'nt':
        path_out = path_in.replace('/','\\')
    else:
        path_out = path_in
    return path_out


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
        checkpoint_folder = make_checkpoint_folder(data_store_folder)
    else:
        # os.mkdir(data_path)
        os.makedirs(data_path,exist_ok=True)
        model_folder = 'model_1'
        data_store_folder = os.path.join(data_path,model_folder)
        os.mkdir(data_store_folder)
        checkpoint_folder = make_checkpoint_folder(data_store_folder)
    return data_store_folder, checkpoint_folder

def make_checkpoint_folder(data_path):
    checkpoint_folder = os.path.join(data_path,'weight_checkpoints')
    os.mkdir(checkpoint_folder)
    return checkpoint_folder

def make_checkpoint_template():
    return "model_weights_epoch_{epoch:02d}.h5"

def get_dir_list(path):
    result = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path,name))]
    return result

def get_completed_model_list(path):
    '''Finds path to folders in directory that have completed training model'''
    result = [name for name in os.listdir(path) if os.path.exists(os.path.join(path,name,f"{name}_results.pkl"))]
    return result

def compare_lists(list1,list2):
    # Find elements in list1 that are not in list2
    unique_elements_in_list1 = list(set(list1) - set(list2))

    # Find elements in list2 that are not in list1
    unique_elements_in_list2 = list(set(list2) - set(list1))

    print("Elements in list1 but not in list2:", unique_elements_in_list1)
    print("Elements in list2 but not in list1:", unique_elements_in_list2)

def record_model_results(output_dir,epochs, batch_size, learning_rate, train_sample_acc, val_sample_acc, test_sample_acc,
                         train_bin_acc, val_bin_acc, test_bin_acc, train_loss, val_loss, test_loss, model, num_train, 
                         num_val, num_test, data_set_name, optimizer_name,start_time,is_shift_data,scale_value):
    # output_path = os.path.join(output_dir,f"results_{df}.txt")
    output_path = make_results_file_name(output_dir,train_sample_acc,val_sample_acc,test_sample_acc)
    with open(output_path,"w") as f:
        formatted_time = get_datetime()
        t_dur = calc_time_duration(start_time,formatted_time)
        f.write(f'{start_time} - {formatted_time}')
        f.write(f'Training Duration = {t_dur}\n')
        f.write(f'trained on file {data_set_name}\n')
        f.write(f'is data shift so course is centered on origin = {is_shift_data}\n')
        f.write(f'Data is scaled by {scale_value}\n')
        f.write('train_sample_acc,val_sample_acc,test_sample_acc,train_bin_acc,val_bin_acc,test_bin_acc,train_loss,val_loss,test_loss,epochs,batch_size,optimizer,learning_rate\n')
        f.write(f'{train_sample_acc},{val_sample_acc},{test_sample_acc},{train_bin_acc},{val_bin_acc},{test_bin_acc},{train_loss:.6f},{val_loss:.6f},{test_loss:.6f},{epochs},{batch_size},{optimizer_name},{learning_rate}\n')
        per_train,per_val,per_test = get_data_percents(num_train,num_val,num_test)
        f.write('percent of data for train, val and test\n')
        f.write(f'percent_train={per_train:.2f},percent_val={per_val:.2f},percent_test={per_test:.2f}\n')
        f.write(f'total number of data points = {num_train + num_test + num_val}\n')
        f.write('num_train_data,num_val_data,num_test_data\n')
        f.write(f'{num_train},{num_val},{num_test}\n')
        write_activation_info_to_file(model,f)
        # following code outputs model summary to file
        sys.stdout = f
        model.summary()
    sys.stdout = sys.__stdout__ #reset stdout to console

def write_activation_info_to_file(model, file_handle):
    """
    Write layer name, layer type, and activation function information to a file.

    Args:
        model: Keras model.
        file_handle: An open file handle where the information will be written.

    Returns:
        None
    """
    file_handle.write("Layer Name, Layer Type, Activation Function\n")
    for layer in model.layers:
        layer_type = type(layer).__name__
        activation_function = layer.get_config().get('activation', 'N/A')
        layer_name = layer.name
        file_handle.write(f"{layer_name}, {layer_type}, {activation_function}\n")

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

def make_output_val_dict(**kwargs):
    '''Function is used in evalaute_keras_model to create the output dictionary from the results if the evaluate code is used in a different script.'''
    return kwargs

## These functions are used in preparing datasets
def get_file_extension(file_path):
    # Get the lowercase file extension
    file_extension = file_path.lower().split('.')[-1]
    
    # Check if the file extension is either "npy" or "csv"
    if file_extension in {'npy', 'csv'}:
        return file_extension
    else:
        raise Exception(f'File path: {file_path} does not end with .npy or .csv')  # Or handle the case when the file has a different extension

def load_data(file_path):
    file_extension = get_file_extension(file_path)
    if file_extension == 'npy':
        return load_np_data(file_path)
    elif file_extension == 'csv':
        return load_csv_data(file_path)
    else:
        raise Exception('Invalid file extension') # I know this is redundant but I am still adding

def load_csv_data(file_path):
    print(f'Loading csv data from {file_path}')
    return np.loadtxt(file_path,delimiter=',')

def load_np_data(file_path):
    print(f'Loading np data from {file_path}')
    return np.load(file_path)

def read_model_num_from_file_path(file_path):
    '''This function can check to see what number is used for a model data path for a single type of obstacle number, to infer the number of obstacles instead of having to specify it each time.
    file_path should be of the form some_path/main_data_file_courses#.csv
    '''
    pattern = r'courses(\d+)[\.csv|\.npy]' #works if fiel is .csv or .npy extension

    # Use re.search to find the pattern in the file path
    match = re.search(pattern, file_path)

    # Check if a match is found
    if match:
        # Extract the matched number from the regex group
        number = match.group(1) # the first group is the entire matched string
        return int(number)
    else:
        raise Exception(f"Model Number not found in string {file_path}.")

def find_null_x_idx(num_obs,max_num_obs):
    '''Returns the indices of the null or filler x obstacles in padded data files.
    num_obs is the number of real obstracles in the file
    max_num _obs is the number of obstacles the file is padded to'''
    start_idx = calc_num_features(num_obs)
    end_idx = calc_num_features(max_num_obs) #constant for the max num obs in padded files
    y_idx = [x for x in range(start_idx,end_idx) if (x-(start_idx))%3 == 0]
    if not len(y_idx) == max_num_obs-num_obs:
        raise Exception(f"Number of y_idx: {len(y_idx)} does not equal num_obs: {max_num_obs-num_obs}") 

    return y_idx

def shuffle_and_split_data(dataset,num_obstacles,split_percentages,shuffle_data=True):
    '''Splits data into train/val and test. Shuffles train/val data, and then splits train and val data.
       INPUTS:
        dataset: file read from load_data
        num_obstacles: number of obstacles in the data file, use size of the padded dataset if using padded data
        split_percentages: length three list indicating the percent for training, validation, and testing
       OUTPUTS:
        split_data: dictionary with train,val,test data as well as opt_costs,num_feat,num_labels'''
    
    if shuffle_data:
        np.random.shuffle(dataset)

    split_data = {}
    total_rows = dataset.shape[0]
    # calcualte num labels and features based on data
    features = calc_num_features(num_obstacles)
    labels = num_obstacles

    # First split into train/val and test without shuffling
    test_split_percent = [split_percentages["train"] + split_percentages["val"], split_percentages["test"]]
    data_splits = split_array(dataset, test_split_percent)

    # Separate into train/val dataset and test dataset
    dataset_TV = data_splits[0]
    dataset_test = data_splits[1]

    # Separate test features and labels and record
    X_test, Y_test = separate_features_labels(dataset_test,num_obstacles)
    split_data["X_test"] = X_test
    split_data["Y_test"] = Y_test

    # shuffle training/validation data
    if shuffle_data:
        np.random.shuffle(dataset_TV)

    # Calculate split percentages based on remaining data
    percent_data_left = 100 - split_percentages["test"]*100
    TV_split_percent = [split_percentages["train"]*100/percent_data_left, split_percentages["val"]*100/percent_data_left] # split percentages for training and validation
    
    # Split data into training/validation, separate features from labels, and record
    TV_data_splits = split_array(dataset_TV,TV_split_percent)
    dataset_train = TV_data_splits[0]
    X_train, Y_train = separate_features_labels(dataset_train,num_obstacles)
    split_data["X_train"] = X_train
    split_data["Y_train"] = Y_train

    dataset_val = TV_data_splits[1]
    X_val, Y_val = separate_features_labels(dataset_val,num_obstacles)
    split_data["X_val"] = X_val
    split_data["Y_val"] = Y_val

    opt_cost_test = dataset_test[:,-1]
    opt_cost_TV = dataset_TV[:,-1]
    opt_costs = np.hstack((opt_cost_TV,opt_cost_test))

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

def combine_test_val_data(split_data):
    '''this combines test and validation data for use in evaluate_keras_model.py.  split_data should be the array created by shuffle_asnd_split_data()'''
    X_train = split_data["X_train"] 
    X_val = split_data["X_val"]   

    Y_train = split_data["Y_train"] 
    Y_val = split_data["Y_val"]   

    X_tv = np.vstack((X_train,X_val))
    Y_tv = np.vstack((Y_train,Y_val))
    return X_tv,Y_tv

def separate_features_labels(dataset,num_obstacles):
    '''Separates data set into features data and label data'''
    num_features = calc_num_features(num_obstacles)
    num_labels = num_obstacles
    feature_data = dataset[:,:num_features]
    label_data = dataset[:,num_features:-1]
    if label_data.shape[1] != num_labels:
        raise Exception(f'incorrect number of labels, expecting {num_labels} but found {label_data.shape[1]}')
    return feature_data, label_data

def calc_num_features(num_obs):
    return 3*num_obs + 4

# def shift_x_axis_row(row,num_obs):


def shift_row(row,num_obs):
    '''shifts data so course is centered at origin (0,0) instead of at (15,15)'''
    interval = 2
    result_list = []

    for ii in range(len(row)):
        if ii > 3 and (ii-4) % (interval + 1) < interval and ii < len(row)-num_obs-1:
            result_list.append(row[ii] - 15)
        elif ii < 4:
            result_list.append(row[ii] - 15)
        else:
            result_list.append(row[ii])
    return result_list

def shift_data_set(data_set,num_obs,is_shift_data):
    '''data set should be an array of read data file'''
    if is_shift_data:
        # implements shifting that data
        modified_array = np.empty_like(data_set)
        for ii,row in enumerate(tqdm(data_set,file=sys.stdout,desc="Shifting Data Set", unit="row")):
            mod_row = shift_row(row,num_obs)
            modified_array[ii,:] = np.array(mod_row)
        return modified_array
    else:
        # returns array unmodified if we dont want to shift data
        print('Data was not shifted.')
        return data_set

def scale_data_set(data_set,num_obs,scale_val,is_scale_data):
    scale = 1/scale_val
    num_features = calc_num_features(num_obs)
    if is_scale_data:
        data_set[:,:num_features] *= scale
        data_set[:,-1] *= scale
        print(f'Data was rescaled by {scale}.')
        return data_set
    else:
        print('Data was not rescaled.')
        return data_set
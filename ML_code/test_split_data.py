import util
import numpy as np


def test_split(file_path):

    dataset = np.loadtxt(file_path,delimiter=',')

    num_obstacles = 3
    # num_obstacles = args.num_obs
    features = 3*num_obstacles + 4
    labels = num_obstacles

    np.random.shuffle(dataset)

    X = dataset[:,:features]
    Y = dataset[:,features:-1]
    if Y.shape[1] != labels:
        raise Exception(f'incorrect number of labels, expecting {labels} but found {Y.shape[1]}')

    opt_costs = dataset[:,-1] # these azre the optimal path costs as found by dijkstra algo during data generation

    split_percentages = [0.7, 0.15, 0.15]

    X_splits = util.split_array(X,split_percentages)
    Y_splits = util.split_array(Y,split_percentages)

    X_train = X_splits[0]
    X_val = X_splits[1]
    X_test = X_splits[2]
    
    Y_train = Y_splits[0]
    Y_val = Y_splits[1]
    Y_test = Y_splits[2]

    for i, split in enumerate(X_splits):
        print(f"Length of Split {i + 1}: {len(split)}")
    
    for i, split in enumerate(Y_splits):
        print(f"Length of Split {i + 1}: {len(split)}")

    print('test done')
    # # split data
    # test_split = 0.8 # percentage to use for training
    # nrows = X.shape[0]
    # split_row = round(test_split*nrows)

    # #TODO make splitting a function
    # X_tv = X[0:split_row,:] # train and validation data
    # Y_tv = Y[0:split_row,:]
    # X_test = X[split_row:,:] # test data
    # Y_test = Y[split_row:,:]

    # nrows = X_tv.shape[0]
    # val_split_row = round(test_split*nrows)

    # X_train = X_tv[0:val_split_row,:]
    # Y_train = Y_tv[0:val_split_row,:]
    # X_val = X_tv[val_split_row:,:]
    # Y_val = Y_tv[val_split_row:,:] 


if __name__ == "__main__":
    file_path = "D:/Vis_network_data/data_file_by_course/main_data_file_courses3.csv"
    # file_path = "./ML_code/Data/small_main_data_file_courses3.csv"
    test_split(file_path)
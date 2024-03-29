import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.python.client import device_lib
from tqdm import tqdm
import util
import util_keras
import graph_util as g_util

def main():

    
    args = util.arg_parse()
    print(device_lib.list_local_devices())

    # user options
    is_shift_data = args.shift # this shifts the data over so that the course is center around origin of (0,0)
    is_scale_data = args.scale_flag
    scale_value = args.scale_value
    # split_percentages = [0.9, 0.05, 0.05]
    # split_percentages = {"train": 0.9, "val": 0.05, "test": 0.05}

    if args.test_file is not None or args.test_idx is not None:
        # ensures that when using these modes, we do not have data get unused as the old test data.
        new_split_percentages = util.change_split_percent_to_TV(args.split_percentages)
    else:
        new_split_percentages = args.split_percentages
    split_percentages = {"train": new_split_percentages[0], "val": new_split_percentages[1], "test": new_split_percentages[2]}
    print(f'using following split percentages: {split_percentages}')
   

    # tf.debugging.set_log_device_placement(True)
 
    # initialize folder for recording results
    data_file = os.path.basename(args.file_path)
    model_output_folder, checkpoint_folder = util.init_data_store_folder(data_file.rstrip('.csv'))

    checkpoint = util_keras.create_checkpoint_callback(checkpoint_folder,util.make_checkpoint_template())

    if args.early_stop:
        # early_stopping = K.callbacks.EarlyStopping(monitor='loss',patience=5,verbose=1,mode='min',restore_best_weights=True)
        # early_stopping = K.callbacks.EarlyStopping(monitor='all_outputs_correct',patience=20,verbose=1,mode='max',restore_best_weights=True)
        early_stopping = K.callbacks.EarlyStopping(monitor='binary_accuracy',patience=3,verbose=1,mode='max',restore_best_weights=True)

    file_path = args.file_path
    tic = time.perf_counter()
    dataset_in = util.load_data(file_path)
    toc = time.perf_counter()
    print(f"Loaded data in {toc - tic:0.4f} seconds")

    #perform data set transformations
    # dataset_processed = util.shift_data_set(dataset_in,args.num_obs,is_shift_d ata)
    # dataset_processed = util.scale_data_set(dataset_processed,args.num_obs,scale_value,is_scale_data)
    dataset_processed = util.transform_data(dataset_in,args.num_obs,is_shift_data,scale_value,is_scale_data)

    # three options depending on how we wish to split the test data
    if args.test_file is not None:
        
        if args.val_file is None:
            # this mode indicates a separate test file is specified and we will want to convert our splits to only be for test and validation
            print(f'loading test data from {args.test_file}.')
            test_split_data = util.load_and_sep_data(args.test_file,args.num_obs,is_shift_data,scale_value,is_scale_data)
            X_test = test_split_data["X"] 
            Y_test = test_split_data["Y"]          
            # now load shuffle and split train and validation data
            split_data = util.shuffle_and_split_data(dataset_processed,args.num_obs,split_percentages,shuffle_data=args.shuffle_tv)
            X_train = split_data["X_train"] 
            Y_train = split_data["Y_train"] 

            X_val = split_data["X_val"]   
            Y_val = split_data["Y_val"] 
        else:
            # This is for the case where 
            print(f'loading test data from {args.test_file}.')
            test_split_data = util.load_and_sep_data(args.test_file,args.num_obs,is_shift_data,scale_value,is_scale_data)
            X_test = test_split_data["X"] 
            Y_test = test_split_data["Y"] 

            val_split_data = util.load_and_sep_data(args.val_file,args.num_obs,is_shift_data,scale_value,is_scale_data)
            X_val = val_split_data["X"]   
            Y_val = val_split_data["Y"] 

            train_split_data = util.separate_train_data(dataset_processed,args.num_obs)
            X_train = train_split_data["X"] 
            Y_train = train_split_data["Y"]

            split_data = util.create_other_parts(args.num_obs) # this is a bandai to get this section running


    elif args.test_idx is not None:
        # this mode will activate if a specific index is specified for where to split the test data from training/validation.  Currently is not implemented.
        print(f'loading test data from index {args.test_idx}')
        split_data = util.test_index_shuffle_and_split_data(dataset_processed,args.num_obs,split_percentages,args.test_idx,shuffle_data=args.shuffle_tv)
        X_test = split_data["X_test"] 
        Y_test = split_data["Y_test"]  

        X_train = split_data["X_train"] 
        Y_train = split_data["Y_train"] 

        X_val = split_data["X_val"]   
        Y_val = split_data["Y_val"] 

    else:
        print('loading test data from split_percentages')
        # If other arguments are not specified then we split data according to percentages in split_percentages
        split_data = util.shuffle_and_split_data(dataset_processed,args.num_obs,split_percentages,shuffle_data=args.shuffle_tv)
        X_test = split_data["X_test"] 
        Y_test = split_data["Y_test"]  

        X_train = split_data["X_train"] 
        Y_train = split_data["Y_train"] 

        X_val = split_data["X_val"]   
        Y_val = split_data["Y_val"] 
      

    opt_costs = split_data["opt_costs"]
    features = split_data["num_features"]
    labels = split_data["num_labels"]
    
    model2test = args.NN_model
    optimizer2use = args.optimizer

    model = K.Sequential()

    
    # attempt for 20 layer model
    # model.add(K.layers.Dense(100, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
    # model.add(K.layers.Dense(200, activation='relu'))
    # # model.add(K.layers.Dense(800, activation='relu'))
    # # model.add(K.layers.Dense(1300, activation='relu'))
    # # model.add(K.layers.Dense(400, activation='relu'))
    # model.add(K.layers.Dense(labels, activation='sigmoid'))

    # attempt for 1 and 2 layer model
    # model.add(K.layers.Dense(12, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
    # model.add(K.layers.Dense(8, activation='relu'))
    # model.add(K.layers.Dense(labels, activation='sigmoid'))

    # attempt for 3 layer model
    # model.add(K.layers.Dense(10, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
    # # model.add(K.layers.Dense(20, activation='relu'))
    # model.add(K.layers.Dense(100, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
    # model.add(K.layers.Dense(100, activation='relu'))
    # model.add(K.layers.Dense(100, activation='relu'))
    # model.add(K.layers.Dense(100, activation='relu'))
    # # model.add(K.layers.Dense(20, activation='relu'))
    # model.add(K.layers.Dense(labels, activation='sigmoid'))

    # model.add(K.layers.Dense(200, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
    # model.add(K.layers.Dense(50, activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
    # model.add(K.layers.Dense(50, activation='relu'))
    # model.add(K.layers.Dense(50, activation='relu'))
    # model.add(K.layers.Dense(50, activation='relu'))
    # model.add(K.layers.Dense(labels, activation='sigmoid'))

    # # same as above but with drop out
    # model.add(K.layers.Dense(200, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
    # model.add(K.layers.Dropout(0.1))
    # model.add(K.layers.Dense(50, activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
    # model.add(K.layers.Dropout(0.1))
    # model.add(K.layers.Dense(50, activation='relu'))
    # model.add(K.layers.Dropout(0.1))
    # model.add(K.layers.Dense(50, activation='relu'))
    # model.add(K.layers.Dropout(0.1))
    # model.add(K.layers.Dense(50, activation='relu'))
    # model.add(K.layers.Dropout(0.1))
    # model.add(K.layers.Dense(labels, activation='sigmoid'))

    # model.add(K.layers.Dense(100, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
    # model.add(K.layers.Dense(100, activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
    # model.add(K.layers.Dense(90, activation='relu'))
    # model.add(K.layers.Dense(90, activation='relu'))
    # model.add(K.layers.Dense(80, activation='relu'))
    # model.add(K.layers.Dense(labels, activation='sigmoid'))

    #  same as above but with dropout
    # model.add(K.layers.Dense(100, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
    # # model.add(K.layers.Dense(20, activation='relu'))
    # model.add(K.layers.Dense(100, activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
    # model.add(K.layers.Dropout(0.1))
    # model.add(K.layers.Dense(90, activation='relu'))
    # model.add(K.layers.Dropout(0.1))
    # model.add(K.layers.Dense(90, activation='relu'))
    # model.add(K.layers.Dropout(0.1))
    # model.add(K.layers.Dense(80, activation='relu'))
    # model.add(K.layers.Dropout(0.1))
    # # model.add(K.layers.Dense(20, activation='relu'))
    # model.add(K.layers.Dense(labels, activation='sigmoid'))

    # model.add(K.layers.Dense(50, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
    # model.add(K.layers.Dense(100, activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
    # model.add(K.layers.Dense(150, activation='sigmoid'))
    # model.add(K.layers.Dense(200, activation='relu'))
    # model.add(K.layers.Dense(250, activation='relu'))
    # model.add(K.layers.Dense(labels, activation='sigmoid'))

    # model.add(K.layers.Dense(250, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
    # model.add(K.layers.Dense(200, activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
    # model.add(K.layers.Dense(150, activation='sigmoid'))
    # model.add(K.layers.Dense(100, activation='relu'))
    # model.add(K.layers.Dense(50, activation='relu'))
    # model.add(K.layers.Dense(labels, activation='sigmoid'))

    # model.add(K.layers.Dense(200, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
    # model.add(K.layers.Dense(150, activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
    # model.add(K.layers.Dense(100, activation='sigmoid'))
    # model.add(K.layers.Dense(150, activation='relu'))
    # model.add(K.layers.Dense(200, activation='relu'))
    # model.add(K.layers.Dense(labels, activation='sigmoid'))

    # model.add(K.layers.Dense(100, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
    # model.add(K.layers.Dense(150, activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
    # model.add(K.layers.Dense(200, activation='sigmoid'))
    # model.add(K.layers.Dense(150, activation='relu'))
    # model.add(K.layers.Dense(100, activation='relu'))
    # model.add(K.layers.Dense(labels, activation='sigmoid'))

    # model.add(K.layers.Dense(300, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
    # # model.add(K.layers.Dense(20, activation='relu'))
    # for ii in range(7):
    #     model.add(K.layers.Dense(280-40*ii, activation='relu'))
    # model.add(K.layers.Dense(20, activation='relu'))
    # model.add(K.layers.Dense(10, activation='sigmoid')) 

    if model2test == 0:
        print('using model 0')
        model.add(K.layers.Dense(200, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
        for ii in range(5):
            model.add(K.layers.Dense(180-40*ii, activation='relu'))
        model.add(K.layers.Dense(labels, activation='tanh'))
        model.add(K.layers.Dense(200, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
        model.add(K.layers.Dense(labels, activation='sigmoid'))

    elif model2test == 1:
        print('using model 1')
        model.add(K.layers.Dense(300, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
        for ii in range(6):
            model.add(K.layers.Dense(260-40*ii, activation='relu'))
        model.add(K.layers.Dense(labels, activation='sigmoid'))
        model.add(K.layers.Dense(200, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
        model.add(K.layers.Dense(labels, activation='sigmoid'))

    elif model2test == 2:
        print('using model 2')
        model.add(K.layers.Dense(200, input_shape=(features,), activation='tanh')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
        model.add(K.layers.Dense(labels, activation='sigmoid'))
    
    elif model2test == 3:
        print('using model 3')
        model.add(K.layers.Dense(500, input_shape=(features,), activation='tanh')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
        model.add(K.layers.Dense(labels, activation='sigmoid'))

    elif model2test == 4:
        # 3 obstacles
        print('using model 4')
        model.add(K.layers.Dense(200, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
        for ii in range(5):
            model.add(K.layers.Dense(180-40*ii, activation='relu'))
        model.add(K.layers.Dense(labels, activation='sigmoid'))
        model.add(K.layers.Dense(200, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
        model.add(K.layers.Dense(labels, activation='sigmoid'))

    elif model2test == 5:
        print('using model 5')
        model.add(K.layers.Dense(200, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
        for ii in range(5):
            model.add(K.layers.Dense(180-40*ii, activation='tanh'))
        model.add(K.layers.Dense(labels, activation='tanh'))
        model.add(K.layers.Dense(200, input_shape=(features,), activation='tanh')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
        model.add(K.layers.Dense(labels, activation='sigmoid'))

    # change to 500 on second to last layer
    elif model2test == 6:
        print('using model 6')
        model.add(K.layers.Dense(500, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
        for ii in range(5):
            model.add(K.layers.Dense(420-80*ii, activation='relu'))
        model.add(K.layers.Dense(labels, activation='sigmoid'))
        model.add(K.layers.Dense(200, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
        model.add(K.layers.Dense(labels, activation='sigmoid'))

    elif model2test == 7:
        print('using model 7')
        first_layer = 620
        neurons_lost_per_layer = 75
        num_hidden_layers = 7
        model = util_keras.create_just_funnel(model,first_layer,neurons_lost_per_layer,num_hidden_layers,features,labels)
        for ii in range(5):
            model.add(K.layers.Dense(105, activation='relu'))
        model.add(K.layers.Dense(labels, activation='sigmoid'))

    elif model2test == 8:
        print('using model 8')
        first_layer = 1000
        neurons_lost_per_layer = 140
        num_hidden_layers = 6
        model = util_keras.create_funnel_model(model,first_layer,neurons_lost_per_layer,num_hidden_layers,features,labels)

    elif model2test == 9:
        print('using model 9')
        first_layer = 620
        neurons_lost_per_layer = 75
        num_hidden_layers = 7
        model = util_keras.create_funnel_model(model,first_layer,neurons_lost_per_layer,num_hidden_layers,features,labels)

    elif model2test == 10:
        print('using model 10')
        first_layer = 1000
        neurons_lost_per_layer = 140
        num_hidden_layers = 6
        model = util_keras.create_double_funnel_model(model,first_layer,neurons_lost_per_layer,num_hidden_layers,features,labels)

    elif model2test == 11:
        print('using model 11')
        first_layer = 1040
        neurons_lost_per_layer = 60
        num_hidden_layers = 16
        model = util_keras.create_funnel_model(model,first_layer,neurons_lost_per_layer,num_hidden_layers,features,labels)

    elif model2test == 12:
        print('using model 12')
        first_layer = 1000
        neurons_lost_per_layer = 140
        num_hidden_layers = 6
        model = util_keras.create_just_funnel(model,first_layer,neurons_lost_per_layer,num_hidden_layers,features,labels)
        model.add(K.layers.Dense(first_layer, activation='sigmoid'))
        model.add(K.layers.Dense(first_layer, activation='sigmoid'))
        model.add(K.layers.Dense(labels, activation='sigmoid'))

    elif model2test == 13:
        print('using model 13')
        first_layer = 1000
        neurons_lost_per_layer = 140
        num_hidden_layers = 6
        model = util_keras.create_just_funnel(model,first_layer,neurons_lost_per_layer,num_hidden_layers,features,labels)
        model.add(K.layers.Dense(55, activation='sigmoid'))
        model.add(K.layers.Dense(55, activation='sigmoid'))
        model.add(K.layers.Dense(labels, activation='sigmoid'))

    elif model2test == 14:
        print('using model 14')
        first_layer = 1560
        neurons_lost_per_layer = 140
        num_hidden_layers = 10
        model = util_keras.create_just_funnel(model,first_layer,neurons_lost_per_layer,num_hidden_layers,features,labels)
        model.add(K.layers.Dense(100, activation='relu'))
        model.add(K.layers.Dense(100, activation='relu'))
        model.add(K.layers.Dense(labels, activation='sigmoid'))

    elif model2test == 15:
        print('using model 15')
        first_layer = 1220
        neurons_lost_per_layer = 150
        num_hidden_layers = 7
        model = util_keras.create_funnel_model(model,first_layer,neurons_lost_per_layer,num_hidden_layers,features,labels)

    elif model2test == 16:
        # for one obstacle
        print('using model 16')
        model.add(K.layers.Dense(12, input_shape=(features,), activation='relu')) 
        model.add(K.layers.Dense(8,activation='relu'))
        model.add(K.layers.Dense(labels, activation='sigmoid'))

    elif model2test == 17:
        # for two obstacle
        print('using model 17')
        model.add(K.layers.Dense(10, input_shape=(features,), activation='relu')) 
        model.add(K.layers.Dense(20,activation='relu'))
        model.add(K.layers.Dense(100,activation='relu'))
        model.add(K.layers.Dense(20,activation='relu'))
        model.add(K.layers.Dense(labels, activation='sigmoid'))
    
    elif model2test == 18:
        print('using model 18')
        model.add(K.layers.Dense(200, input_shape=(features,), activation='sigmoid')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
        model.add(K.layers.Dense(labels, activation='sigmoid'))

    elif model2test == 19:
        print('using model 19')
        model.add(K.layers.Dense(200, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
        model.add(K.layers.Dense(labels, activation='sigmoid'))

    elif model2test == 20:
        print('using model 20')
        model.add(K.layers.Dense(200, input_shape=(features,), activation='tanh')) 
        model.add(K.layers.Dense(200, activation='tanh')) 
        model.add(K.layers.Dense(labels, activation='sigmoid'))

    elif model2test == 21:
        print('using model 21')
        model.add(K.layers.Dense(300, input_shape=(features,), activation='tanh')) 
        model.add(K.layers.Dense(labels, activation='sigmoid'))

    elif model2test == 22:
        print('using model 22')
        model.add(K.layers.Dense(100, input_shape=(features,), activation='relu')) 
        model.add(K.layers.Dense(100, activation='relu')) 
        model.add(K.layers.Dense(100, activation='relu')) 
        model.add(K.layers.Dense(100, activation='relu')) 
        model.add(K.layers.Dense(labels, activation='sigmoid'))

    elif model2test == 23:
        print('using model 23')
        model.add(K.layers.Dense(100, input_shape=(features,), activation='tanh')) 
        model.add(K.layers.Dense(labels, activation='sigmoid'))

    else:
        print('using default model')
        model.add(K.layers.Dense(50, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
        model.add(K.layers.Dense(100, activation='relu'))
        model.add(K.layers.Dense(100, activation='relu'))
        model.add(K.layers.Dense(100, activation='relu'))
        model.add(K.layers.Dense(labels, activation='sigmoid'))

    # model.add(K.layers.Dense(200, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
    # # model.add(K.layers.Dense(20, activation='relu'))
    # for ii in range(9):
    #     model.add(K.layers.Dense(180-20*ii, activation='relu'))
    # # model.add(K.layers.Dense(20, activsation='relu'))
    # model.add(K.layers.Dense(labels, activation='sigmoid'))
    # model.add(K.layers.Dense(200, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
    # model.add(K.layers.Dense(labels, activation='sigmoid'))
   
    # model.add(K.layers.Dense(50, input_shape=(features,), activation='tanh')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
    # model.add(K.layers.Dense(50, activation='tanh')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
    # model.add(K.layers.Dense(50, activation='tanh'))
    # model.add(K.layers.Dense(50, activation='tanh'))
    # model.add(K.layers.Dense(50, activation='tanh'))
    # model.add(K.layers.Dense(labels, activation='sigmoid'))

    # model.add(K.layers.Dense(50, input_shape=(features,), activation='sigmoid')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
    # model.add(K.layers.Dense(50, activation='sigmoid')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
    # model.add(K.layers.Dense(50, activation='sigmoid'))
    # model.add(K.layers.Dense(50, activation='sigmoid'))
    # model.add(K.layers.Dense(50, activation='sigmoid'))
    # model.add(K.layers.Dense(labels, activation='sigmoid'))
    
   
    # model.add(K.layers.Dense(10, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
    # model.add(K.layers.Dense(20, activation='relu'))
    # model.add(K.layers.Dense(100, activation='relu'))
    # model.add(K.layers.Dense(20, activation='relu'))
    # model.add(K.layers.Dense(labels, activation='sigmoid'))
    print(f'Number of Features = {features}')
    model.summary()

    # compile the keras model
    learning_rate = args.learning_rate
    optimizer = util_keras.optimizer_selector(optimizer2use, learning_rate)
    
    # optimizer = K.optimizers.RMSprop(learning_rate=learning_rate)
    # optimizer = K.optimizers.Adam()
    # model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy',util_keras.AllOutputsCorrect()])

    # fit the keras model on the dataset
    n_epochs = args.n_epochs
    b_size = args.batch_size
    start_time = util.get_datetime(add_new_line=False)
    try:
        if args.early_stop:
            results = model.fit(X_train, Y_train, validation_data = (X_val,Y_val), epochs=n_epochs, batch_size=b_size, callbacks=[checkpoint, early_stopping], shuffle = True)
        else:
            results = model.fit(X_train, Y_train, validation_data = (X_val,Y_val), epochs=n_epochs, batch_size=b_size, callbacks=[checkpoint], shuffle = True)
        
    # except KeyboardInterrupt:
    except:
        #save intermediate results upon exception
        # train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy = util_keras.evaluate_model(X_train, X_val, X_test, Y_train, Y_val, Y_test, model, b_size) #old way
        train_loss,train_bin_acc,train_sample_acc,val_loss,val_bin_acc,val_sample_acc,test_loss,test_bin_acc,test_sample_acc = util_keras.evaluate_model(X_train, X_val, X_test, Y_train, Y_val, Y_test, model, b_size)
        model.save(os.path.join(model_output_folder,"keras_model"))
        _, f_trained = os.path.split(args.file_path)
        # util.record_model_results(model_output_folder,n_epochs,b_size,learning_rate,train_accuracy*100,
        #                       val_accuracy*100,test_accuracy*100,train_loss,val_loss,test_loss,
        #                       model,X_train.shape[0],X_val.shape[0],X_test.shape[0],f_trained,
        #                       optimizer._name,start_time,is_shift_data,scale_value) 
        util.record_model_results(model_output_folder,n_epochs,b_size,learning_rate,train_sample_acc*100,
                              val_sample_acc*100,test_sample_acc*100,train_bin_acc*100, val_bin_acc*100,
                              test_bin_acc*100,train_loss,val_loss,test_loss,model,X_train.shape[0],
                              X_val.shape[0],X_test.shape[0],f_trained,optimizer._name,start_time,is_shift_data,scale_value) 
        return    
    
    # evaluate the keras model
    # train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy = util_keras.evaluate_model(X_train, X_val, X_test, Y_train, Y_val, Y_test, model, b_size)
    train_loss,train_bin_acc,train_sample_acc,val_loss,val_bin_acc,val_sample_acc,test_loss,test_bin_acc,test_sample_acc = util_keras.evaluate_model(X_train, X_val, X_test, Y_train, Y_val, Y_test, model, b_size)

    # record model training results
    model.save(os.path.join(model_output_folder,"keras_model"))
    _, f_trained = os.path.split(args.file_path)
    util.record_model_results(model_output_folder,n_epochs,b_size,learning_rate,train_sample_acc*100,
                              val_sample_acc*100,test_sample_acc*100,train_bin_acc*100, val_bin_acc*100,
                              test_bin_acc*100,train_loss,val_loss,test_loss,model,X_train.shape[0],
                              X_val.shape[0],X_test.shape[0],f_trained,optimizer._name,start_time,is_shift_data,scale_value) 
    util.record_model_fit_results(results,model_output_folder)
    g_util.save_loss_acc_plot(results.history,model_output_folder)
    print('\ntraining complete')




if __name__ == "__main__":
    main()

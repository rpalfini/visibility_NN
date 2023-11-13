import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.python.client import device_lib
from tqdm import tqdm
import util
import util_keras

def main():

    
    args = util.arg_parse()
    print(device_lib.list_local_devices())

    # user options
    is_shift_data = args.shift # this shifts the data over so that the course is center around origin of (0,0)
    # split_percentages = [0.9, 0.05, 0.05]
    split_percentages = {"train": 0.9, "val": 0.05, "test": 0.05}
   

    # tf.debugging.set_log_device_placement(True)

    # initialize folder for recording results
    data_file = os.path.basename(args.file_path)
    model_output_folder, checkpoint_folder = util.init_data_store_folder(data_file.rstrip('.csv'))

    checkpoint = util_keras.create_checkpoint_callback(checkpoint_folder,util.make_checkpoint_template())

    file_path = args.file_path
    tic = time.perf_counter()
    dataset_in = util.load_data(file_path)
    toc = time.perf_counter()
    print(f"Loaded data in {toc - tic:0.4f} seconds")

    
    dataset_processed = util.shift_data_set(dataset_in,args.num_obs,is_shift_data)
    split_data = util.shuffle_and_split_data(dataset_processed,args.num_obs,split_percentages)

    X_train = split_data["X_train"] 
    X_val = split_data["X_val"]   
    X_test = split_data["X_test"]  

    Y_train = split_data["Y_train"] 
    Y_val = split_data["Y_val"]   
    Y_test = split_data["Y_test"]  

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

    # change to 500 on second to last layer
    elif model2test == 7:
        print('using model 7')
        model.add(K.layers.Dense(500, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
        for ii in range(11):
            model.add(K.layers.Dense(460-40*ii, activation='relu'))
        model.add(K.layers.Dense(labels, activation='sigmoid'))
        model.add(K.layers.Dense(500, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
        model.add(K.layers.Dense(labels, activation='sigmoid'))

    elif model2test == 8:
        print('using model 8')
        first_layer = 1000
        neurons_lost_per_layer = 140
        num_hidden_layers = 6
        model = util_keras.create_funnel_model(model,first_layer,neurons_lost_per_layer,num_hidden_layers,features,labels)

    elif model2test == 9:
        print('using model 9')
        first_layer = 3220
        neurons_lost_per_layer = 200
        num_hidden_layers = 15
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
    if optimizer2use == 0:
        optimizer = K.optimizers.Adam(learning_rate=learning_rate)
        print('optimizer is Adam')
    elif optimizer2use == 1:
        optimizer = K.optimizers.RMSprop(learning_rate=learning_rate)
        print('optimizer is RMSprop')
    elif optimizer2use == 2:
        optimizer = K.optimizers.SGD(learning_rate=learning_rate)
        print('optimizer is SGD')
    else:
        optimizer = K.optimizers.Adam(learning_rate=learning_rate)
        print('optimizer is Adam')
    
    # optimizer = K.optimizers.RMSprop(learning_rate=learning_rate)
    # optimizer = K.optimizers.Adam()
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # fit the keras model on the dataset
    n_epochs = args.n_epochs
    b_size = args.batch_size
    start_time = util.get_datetime(add_new_line=False)
    try:
        results = model.fit(X_train, Y_train, validation_data = (X_val,Y_val), epochs=n_epochs, batch_size=b_size, callbacks=[checkpoint],shuffle = True)
        
    # except KeyboardInterrupt:
    except:
        #save intermediate results upon exception
        train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy = util_keras.evaluate_model(X_train, X_val, X_test, Y_train, Y_val, Y_test, model, b_size)
        model.save(os.path.join(model_output_folder,"keras_model"))
        _, f_trained = os.path.split(args.file_path)
        # util.record_model_results(model_output_folder,n_epochs,b_size,learning_rate,0,
        #                         0,0,0,0,0,model,X_train.shape[0],X_val.shape[0],
        #                         X_test.shape[0],f_trained,optimizer._name,start_time,is_shift_data)
        util.record_model_results(model_output_folder,n_epochs,b_size,learning_rate,train_accuracy*100,
                              val_accuracy*100,test_accuracy*100,train_loss,val_loss,test_loss,
                              model,X_train.shape[0],X_val.shape[0],X_test.shape[0],f_trained,
                              optimizer._name,start_time,is_shift_data) 
        return    
    
    # evaluate the keras model
    train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy = util_keras.evaluate_model(X_train, X_val, X_test, Y_train, Y_val, Y_test, model, b_size)

    # record model training results
    model.save(os.path.join(model_output_folder,"keras_model"))
    _, f_trained = os.path.split(args.file_path)
    util.record_model_results(model_output_folder,n_epochs,b_size,learning_rate,train_accuracy*100,
                              val_accuracy*100,test_accuracy*100,train_loss,val_loss,test_loss,
                              model,X_train.shape[0],X_val.shape[0],X_test.shape[0],f_trained,
                              optimizer._name,start_time,is_shift_data)
    util.record_model_fit_results(results,model_output_folder)
    print('\ntraining complete')


if __name__ == "__main__":
    main()

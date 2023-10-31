import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.python.client import device_lib
import util
import util_keras
import os

def main():

    start_time = util.get_datetime(add_new_line=False)
    args = util.arg_parse()
    print(device_lib.list_local_devices())

    # tf.debugging.set_log_device_placement(True)

    # initialize folder for recording results
    data_file = os.path.basename(args.file_path)
    model_output_folder, checkpoint_folder = util.init_data_store_folder(data_file.rstrip('.csv'))

    checkpoint = util_keras.create_checkpoint_callback(checkpoint_folder,util.make_checkpoint_template())

    file_path = args.file_path
    split_percentages = [0.9, 0.05, 0.05]
    dataset = util.load_data(file_path)
    split_data = util.shuffle_and_split_data(dataset,args.num_obs,split_percentages)

    X_train = split_data["X_train"] 
    X_val = split_data["X_val"]   
    X_test = split_data["X_test"]  

    Y_train = split_data["Y_train"] 
    Y_val = split_data["Y_val"]   
    Y_test = split_data["Y_test"]  

    opt_costs = split_data["opt_costs"]
    features = split_data["num_features"]
    labels = split_data["num_labels"]
    

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
    # model.add(K.layers.Dense(100, activation='relu'))
    # model.add(K.layers.Dense(100, activation='relu'))
    # model.add(K.layers.Dense(100, activation='relu'))
    # # model.add(K.layers.Dense(20, activation='relu'))
    # model.add(K.layers.Dense(labels, activation='sigmoid'))

    model.add(K.layers.Dense(200, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
    # model.add(K.layers.Dense(20, activation='relu'))
    for ii in range(9):
        model.add(K.layers.Dense(190-20*ii, activation='relu'))
    # model.add(K.layers.Dense(20, activation='relu'))
    model.add(K.layers.Dense(labels, activation='sigmoid'))

    # model.add(K.layers.Dense(10, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
    # model.add(K.layers.Dense(20, activation='relu'))
    # model.add(K.layers.Dense(100, activation='relu'))
    # model.add(K.layers.Dense(20, activation='relu'))
    # model.add(K.layers.Dense(labels, activation='sigmoid'))
    print(f'Number of Features = {features}')
    model.summary()

    # compile the keras model
    learning_rate = args.learning_rate
    optimizer = K.optimizers.Adam(learning_rate=learning_rate)
    # optimizer = K.optimizers.Adam()
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # fit the keras model on the dataset
    n_epochs = args.n_epochs
    b_size = args.batch_size
    results = model.fit(X_train, Y_train, validation_data = (X_val,Y_val), epochs=n_epochs, batch_size=b_size, callbacks=[checkpoint])

    # evaluate the keras model
    print('testing training data')
    _, train_accuracy = model.evaluate(X_train, Y_train)
    print('testing validation data')
    _, val_accuracy = model.evaluate(X_val, Y_val)
    print('testing test data')
    _, test_accuracy = model.evaluate(X_test, Y_test)
    print('Train_Accuracy: %.2f' % (train_accuracy*100))
    print('Validation_Accuracy: %.2f' % (val_accuracy*100))
    print('Test_Accuracy: %.2f' % (test_accuracy*100))
    
    # record model training results
    # data_file = os.path.basename(file_path)
    # model_output_folder = util.init_data_store_folder(data_file.rstrip('.csv'))
    model.save(os.path.join(model_output_folder,"keras_model"))
    _, f_trained = os.path.split(args.file_path)
    util.record_model_results(model_output_folder,n_epochs,b_size,learning_rate,train_accuracy*100,val_accuracy*100,test_accuracy*100,model,X_train.shape[0],X_val.shape[0],X_test.shape[0],f_trained,optimizer._name,start_time)
    util.record_model_fit_results(results,model_output_folder)
    print('training complete')

if __name__ == "__main__":
    main()

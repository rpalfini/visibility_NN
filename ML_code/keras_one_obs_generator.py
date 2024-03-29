import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.python.client import device_lib
import util
import csv_file_combiner as cfc

print(device_lib.list_local_devices())

# data_folder = './results_merge/'
# # data_folder = 'H:/My Drive/Visibility_data_generation/Data Backups/23_02_18_and_19/'
# # data_file = '23_02_18_batch2_2_course_18_obs_data.csv'
# data_file = '23_02_18_19_20_merge_fixed.csv'
# # data_file = 'test_file_fixed.csv'
# # data_file = '23_02_18_and_19_merge.csv'
# dataset = np.loadtxt(data_folder+data_file,delimiter=',')

# num_obstacles = 20
# features = 3*num_obstacles + 4
# labels = num_obstacles

# np.random.shuffle(dataset)

# X = dataset[:,:features]
# Y = dataset[:,features:-1]
# if Y.shape[1] != labels:
#     raise Exception('incorrect number of labels')

# opt_costs = dataset[:,-1]

# # split data
# test_split = 0.8 # percentage to use for training
# nrows = X.shape[0]
# split_row = round(test_split*nrows)

# #TODO make splitting a function
# X_tv = X[0:split_row,:] # train and validation data
# Y_tv = Y[0:split_row,:]
# X_test = X[split_row:,:]
# Y_test = Y[split_row:,:]

# nrows = X_tv.shape[0]
# val_split_row = round(test_split*nrows)

# X_train = X_tv[0:val_split_row,:]
# Y_train = Y_tv[0:val_split_row,:]
# X_val = X_tv[val_split_row:,:]
# Y_val = Y_tv[val_split_row:,:]






params = {'dim': (32,32,32),
          'batch_size': 64,
          'n_classes': 2,
          'n_channels': 1,
          'shuffle': True}



model = K.Sequential()
model.add(K.layers.Dense(100, input_shape=(64,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
model.add(K.layers.Dense(200, activation='relu'))
# model.add(K.layers.Dense(800, activation='relu'))
# model.add(K.layers.Dense(1300, activation='relu'))
# model.add(K.layers.Dense(400, activation='relu'))
model.add(K.layers.Dense(20, activation='sigmoid'))

# compile the keras model
optimizer = K.optimizers.Adam(lr=0.0001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# fit the keras model on the dataset
n_epochs = 100
b_size = 64
results = model.fit(X_train, Y_train, validation_data = (X_val,Y_val), epochs=n_epochs, batch_size=b_size)

# evaluate the keras model
_, train_accuracy = model.evaluate(X_train, Y_train)
_, test_accuracy = model.evaluate(X_test, Y_test)
print('Train_Accuracy: %.2f' % (train_accuracy*100))
print('Test_Accuracy: %.2f' % (test_accuracy*100))
# model.save('C:/Users/Robert/git/visibility_NN')
model_output_folder = util.init_data_store_folder(data_file.strip('.csv'))
model.save(model_output_folder+"\keras_model")
util.record_model_results(model_output_folder,n_epochs,b_size,train_accuracy*100,test_accuracy*100)
util.record_model_fit_results(results,model_output_folder)

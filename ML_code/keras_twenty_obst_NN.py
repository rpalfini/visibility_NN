import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.python.client import device_lib
import util
import os

args = util.arg_parse()

print(device_lib.list_local_devices())

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# data_folder = 'G:/My Drive/Python/Visual Graph'

# dataset = np.loadtxt(data_folder+'/2022_10_17one_obst data_large.csv',delimiter=',')
# tf.debugging.set_log_device_placement(True)

# data_folder = './ML_code/Data'
# data_folder = 'D:\Vis_network_data\data_file_by_course'
# # data_folder = './results_merge/'
# # data_folder = 'H:/My Drive/Visibility_data_generation/Data Backups/23_02_18_and_19/'
# # data_file = '23_02_18_batch2_2_course_18_obs_data.csv'
# # data_file = '23_02_18_19_20_merge_fixed.csv'
# data_file = 'main_data_file_courses1.csv'
# data_file = 'test_file_fixed.csv'
# data_file = '23_02_18_and_19_merge.csv'
# file_path = os.path.join(data _folder,data_file)
file_path = args.file_path
dataset = np.loadtxt(file_path,delimiter=',')

# num_obstacles = 3
num_obstacles = args.num_obs
features = 3*num_obstacles + 4
labels = num_obstacles

np.random.shuffle(dataset)

X = dataset[:,:features]
Y = dataset[:,features:-1]
if Y.shape[1] != labels:
    raise Exception('incorrect number of labels')

opt_costs = dataset[:,-1]

# split data
test_split = 0.8 # percentage to use for training
nrows = X.shape[0]
split_row = round(test_split*nrows)

#TODO make splitting a function
X_tv = X[0:split_row,:] # train and validation data
Y_tv = Y[0:split_row,:]
X_test = X[split_row:,:]
Y_test = Y[split_row:,:]

nrows = X_tv.shape[0]
val_split_row = round(test_split*nrows)

X_train = X_tv[0:val_split_row,:]
Y_train = Y_tv[0:val_split_row,:]
X_val = X_tv[val_split_row:,:]
Y_val = Y_tv[val_split_row:,:]

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
model.add(K.layers.Dense(10, input_shape=(features,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
model.add(K.layers.Dense(20, activation='relu'))
model.add(K.layers.Dense(20, activation='relu'))
model.add(K.layers.Dense(labels, activation='sigmoid'))

# compile the keras model
# optimizer = K.optimizers.Adam(learning_rate=0.0001)
optimizer = K.optimizers.Adam()
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# fit the keras model on the dataset
n_epochs = args.n_epochs
b_size = args.batch_size
results = model.fit(X_train, Y_train, validation_data = (X_val,Y_val), epochs=n_epochs, batch_size=b_size)

# evaluate the keras model
_, train_accuracy = model.evaluate(X_train, Y_train)
_, test_accuracy = model.evaluate(X_test, Y_test)
print('Train_Accuracy: %.2f' % (train_accuracy*100))
print('Test_Accuracy: %.2f' % (test_accuracy*100))
# model.save('C:/Users/Robert/git/visibility_NN')
data_file = os.path.basename(file_path)
model_output_folder = util.init_data_store_folder(data_file.strip('.csv'))
model.save(model_output_folder+"\keras_model")
util.record_model_results(model_output_folder,n_epochs,b_size,train_accuracy*100,test_accuracy*100)
util.record_model_fit_results(results,model_output_folder)

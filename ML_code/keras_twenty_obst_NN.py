import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.python.client import device_lib
import util

print(device_lib.list_local_devices())

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# data_folder = 'G:/My Drive/Python/Visual Graph'

# dataset = np.loadtxt(data_folder+'/2022_10_17one_obst data_large.csv',delimiter=',')
# tf.debugging.set_log_device_placement(True)


data_folder = 'C:/Users/Robert/git/visibility_NN/results_merge/'
# data_file = '23_02_18_batch2_2_course_18_obs_data.csv'
data_file = '23_02_18_merge.csv'
# data_file = '23_02_18_and_19_merge.csv'
dataset = np.loadtxt(data_folder+data_file,delimiter=',')

num_obstacles = 20
features = 3*num_obstacles + 4
labels = num_obstacles

X = dataset[:,:features]
Y = dataset[:,features:-1]
if Y.shape[1] != labels:
    raise Exception('incorrect number of labels')

opt_costs = dataset[:,-1]

# split data
train_split = 0.8 # percentage to use for training
nrows = X.shape[0]
split_row = round(train_split*nrows)

X_train = X[0:split_row,:]
Y_train = Y[0:split_row]
X_val = X[split_row:,:]
Y_val = Y[split_row:]

model = K.Sequential()
model.add(K.layers.Dense(100, input_shape=(64,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
model.add(K.layers.Dense(150, activation='relu'))
model.add(K.layers.Dense(100, activation='relu'))
model.add(K.layers.Dense(20, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
n_epochs = 100
b_size = 32
model.fit(X_train, Y_train, epochs=n_epochs, batch_size=b_size)

# evaluate the keras model
_, train_accuracy = model.evaluate(X_train, Y_train)
_, val_accuracy = model.evaluate(X_val, Y_val)
print('Train_Accuracy: %.2f' % (train_accuracy*100))
print('Val_Accuracy: %.2f' % (val_accuracy*100))
# model.save('C:/Users/Robert/git/visibility_NN')
model_output_folder = util.init_data_store_folder(data_file.strip('.csv'))
model.save(model_output_folder)
util.record_model_results(model_output_folder,n_epochs,b_size,train_accuracy*100,val_accuracy*100)
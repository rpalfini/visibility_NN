import numpy as np
import tensorflow as tf
from tensorflow import keras as K
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# data_folder = 'G:/My Drive/Python/Visual Graph'

# dataset = np.loadtxt(data_folder+'/2022_10_17one_obst data_large.csv',delimiter=',')

data_folder = 'C:/Users/Robert/git/visibility_NN/results_merge'

dataset = np.loadtxt(data_folder+'/22_10_17_one_obstacle_data_merge.csv',delimiter=',')

X = dataset[:,0:-1]
Y = dataset[:,-1]

# split data
train_split = 0.8
nrows = X.shape[0]
split_row = round(train_split*nrows)

X_train = X[0:split_row,:]
Y_train = Y[0:split_row,:]
X_val = X[split_row:,:]
Y_val = Y[split_row:,:]

model = K.Sequential()
model.add(K.layers.Dense(12, input_shape=(7,), activation='relu')) #specify shape of input layer to match number of features.  This is done on the first hidden layer.
model.add(K.layers.Dense(8, activation='relu'))
model.add(K.layers.Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X, Y, epochs=20, batch_size=10)

# evaluate the keras model
_, train_accuracy = model.evaluate(X_train, Y_train)
_, val_accuracy = model.evaluate(X_val, Y_val)
print('Train_Accuracy: %.2f' % (train_accuracy*100))
print('Val_Accuracy: %.2f' % (val_accuracy*100))
model.save('C:/Users/Robert/git/visibility_NN/results_merge')
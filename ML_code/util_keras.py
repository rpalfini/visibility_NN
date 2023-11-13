import os
from keras.callbacks import ModelCheckpoint
from tensorflow import keras as K

# This file is for utility functions that require keras library
def create_checkpoint_callback(subdirectory, filename_template):
    checkpoint = ModelCheckpoint(
        os.path.join(subdirectory, filename_template),
        save_weights_only=True,
        save_freq="epoch"
    )
    return checkpoint
 
def evaluate_model(X_train, X_val, X_test, Y_train, Y_val, Y_test, model, batch_size):
    print('testing training data')
    train_loss, train_accuracy = model.evaluate(X_train,Y_train, batch_size = batch_size)
    print('testing validation data')
    val_loss, val_accuracy = model.evaluate(X_val, Y_val, batch_size = batch_size)
    print('testing test data')
    test_loss, test_accuracy = model.evaluate(X_test, Y_test, batch_size = batch_size)
    print('\nTrain_Accuracy: %.2f' % (train_accuracy*100))
    print('Validation_Accuracy: %.2f' % (val_accuracy*100))
    print('Test_Accuracy: %.2f' % (test_accuracy*100))
    print('\nTrain_Loss: %.6f' % (train_loss))
    print('Validation_Loss: %.6f' % (val_loss))
    print('Test_Loss: %.6f' % (test_loss))
    return train_loss,train_accuracy,val_loss,val_accuracy,test_loss,test_accuracy

def create_funnel_model(model,first_layer, neurons_lost_per_layer, num_hidden_layers,features,labels):
    '''creates specific neural network model based on inputs.  This one has worked fairly well.'''
    model.add(K.layers.Dense(first_layer, input_shape=(features,), activation='relu')) 
    for ii in range(num_hidden_layers):
        model.add(K.layers.Dense(first_layer-neurons_lost_per_layer-neurons_lost_per_layer*ii, activation='relu'))
    model.add(K.layers.Dense(labels, activation='sigmoid'))
    model.add(K.layers.Dense(first_layer, activation='relu')) 
    model.add(K.layers.Dense(labels, activation='sigmoid'))
    return model

def create_double_funnel_model(model,first_layer, neurons_lost_per_layer, num_hidden_layers,features,labels):
    '''creates specific neural network model based on inputs.  This one has worked fairly well.'''
    model.add(K.layers.Dense(first_layer, input_shape=(features,), activation='relu')) 
    for ii in range(num_hidden_layers):
        model.add(K.layers.Dense(first_layer-neurons_lost_per_layer-neurons_lost_per_layer*ii, activation='relu'))
    model.add(K.layers.Dense(labels, activation='sigmoid'))
    model.add(K.layers.Dense(first_layer, activation='relu')) 
    model.add(K.layers.Dense(labels, activation='sigmoid'))
    model.add(K.layers.Dense(first_layer, activation='relu')) 
    model.add(K.layers.Dense(labels, activation='sigmoid'))
    return model
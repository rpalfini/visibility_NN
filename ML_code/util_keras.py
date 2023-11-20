import os
from keras.callbacks import ModelCheckpoint
from tensorflow import keras as K
import tensorflow as tf

# This file is for utility classes and functions that require keras library

# These class and functions are used during training and evaluation.

class AllOutputsCorrect(tf.keras.metrics.Metric):
    def __init__(self, name='all_outputs_correct', **kwargs):
        super(AllOutputsCorrect, self).__init__(name=name, **kwargs)
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros')
        self.correct_samples = self.add_weight(name='correct_samples', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        dbg = False
        # Assuming y_true and y_pred are arrays with binary values (0 or 1)
        # Threshold for predicted probabilities (e.g., 0.5 for sigmoid activation)
        threshold = 0.5
        if dbg:
            print(f"y_true:{y_true}")
            print(f"y_pred:{y_pred}")
        # Determine predicted classes
        y_pred_classes = tf.cast(y_pred > threshold, tf.float32)
        
        y_true = tf.cast(y_true, tf.float32)
        if dbg:
            print(f"y_pred_classes:{y_pred_classes}")
            print(f"y_true:{y_true}")
        # Compare predicted classes to true classes for all outputs
        all_outputs_correct = tf.reduce_all(tf.equal(y_pred_classes, y_true), axis=-1)
        if dbg:
            print(f"tf.equal(y_pred_classes, y_true): {tf.equal(y_pred_classes, y_true)}")
            print(f"all_outputs_correct: {all_outputs_correct}")
            print(f"tf_shape(y_true):{tf.shape(y_true)[0]}")
            print(f"tf.cast(tf.shape(y_true)[0], tf.float32): {tf.cast(tf.shape(y_true)[0], tf.float32)}")
            print(f"tf.cast(tf.shape(y_true), tf.float32): {tf.cast(tf.shape(y_true), tf.float32)}")
            print(f"tf.reduce_sum(tf.cast(all_outputs_correct, tf.float32)):{tf.reduce_sum(tf.cast(all_outputs_correct, tf.float32))}")
            print(f"len(y_true.shape):{len(y_true.shape)}")
        # Update total and correct samples
        self.total_samples.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

        self.correct_samples.assign_add(tf.reduce_sum(tf.cast(all_outputs_correct, tf.float32)))
        if dbg:
            print(f"total_samples: {self.total_samples}")
            print(f"correct_samples: {self.correct_samples}")

    def result(self):
        # Calculate and return the proportion of samples where all outputs were correct
        dbg = False
        if dbg:
            print(f'sample accuracy: {self.correct_samples / self.total_samples}')
        return self.correct_samples / self.total_samples

def optimizer_selector(optimizer2use, learning_rate):
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
    return optimizer

def create_checkpoint_callback(subdirectory, filename_template):
    checkpoint = ModelCheckpoint(
        os.path.join(subdirectory, filename_template),
        save_weights_only=True,
        save_freq="epoch"
    )
    return checkpoint

def print_metric_results():
    pass

def evaluate_model_old(X_train, X_val, X_test, Y_train, Y_val, Y_test, model, batch_size):
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

#following function will be used in keras_twenty_obs_NN.py
def evaluate_model(X_train, X_val, X_test, Y_train, Y_val, Y_test, model, batch_size):
    print('testing training data')
    train_loss, train_bin_acc, train_sample_acc = model.evaluate(X_train,Y_train, batch_size = batch_size)
    print('testing validation data')
    val_loss, val_bin_acc, val_sample_acc = model.evaluate(X_val, Y_val, batch_size = batch_size)
    print('testing test data')
    test_loss, test_bin_acc, test_sample_acc = model.evaluate(X_test, Y_test, batch_size = batch_size)
    print_bin_acc_results(train_bin_acc, val_bin_acc, test_bin_acc)
    print_sample_acc_results(train_sample_acc, val_sample_acc, test_sample_acc)
    print_loss_results(train_loss, val_loss, test_loss)
    return train_loss,train_bin_acc,train_sample_acc,val_loss,val_bin_acc,val_sample_acc,test_loss,test_bin_acc,test_sample_acc

def print_loss_results(train_loss, val_loss, test_loss):
    print('\nTrain_Loss: %.6f' % (train_loss))
    print('Validation_Loss: %.6f' % (val_loss))
    print('Test_Loss: %.6f' % (test_loss))

def print_sample_acc_results(train_sample_acc, val_sample_acc, test_sample_acc):
    print('\nSample_Train_Accuracy: %.2f' % (train_sample_acc*100))
    print('Sample_Validation_Accuracy: %.2f' % (val_sample_acc*100))
    print('Sample_Test_Accuracy: %.2f' % (test_sample_acc*100))

def print_bin_acc_results(train_bin_acc, val_bin_acc, test_bin_acc):
    print('\nTrain_Bin_Accuracy: %.2f' % (train_bin_acc*100))
    print('Validation_Bin_Accuracy: %.2f' % (val_bin_acc*100))
    print('Test_Bin_Accuracy: %.2f' % (test_bin_acc*100))

# These functions are used to load models when evaluating previously trained models.


# These functions create NN model architectures using parameters
def create_funnel_model(model,first_layer, neurons_lost_per_layer, num_hidden_layers,features,labels):
    '''creates specific neural network model based on inputs.  This one has worked fairly well.'''
    model.add(K.layers.Dense(first_layer, input_shape=(features,), activation='relu')) 
    for ii in range(num_hidden_layers):
        model.add(K.layers.Dense(first_layer-neurons_lost_per_layer-neurons_lost_per_layer*ii, activation='relu'))
    model.add(K.layers.Dense(labels, activation='sigmoid'))
    model.add(K.layers.Dense(first_layer, activation='relu')) 
    model.add(K.layers.Dense(labels, activation='sigmoid'))
    return model

def create_just_funnel(model,first_layer, neurons_lost_per_layer, num_hidden_layers,features,labels):
    '''creates specific neural network model based on inputs. First half of the one that has worked well'''
    model.add(K.layers.Dense(first_layer, input_shape=(features,), activation='relu')) 
    for ii in range(num_hidden_layers):
        model.add(K.layers.Dense(first_layer-neurons_lost_per_layer-neurons_lost_per_layer*ii, activation='relu'))
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
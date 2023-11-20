import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tensorflow import keras as K
import util
import util_keras


def main(model_path,epoch,data_file,num_obs,batch,is_shift_data):
    # This function lets us load a model with specific weights and evaluate on a new dataset
    if epoch is None:
        weight_loaded_model, _ = load_model(model_path)
    else:
        weight_loaded_model = load_model_with_checkpoint(model_path,epoch)

    is_retroactive_test = True  # Adding this option to specify if we are testing custom metric on previously trained model where we need to recompile with the new metric.  Any models created prior to 11/18 at 6:30pm require retroactive tests.

    if is_retroactive_test:
        # new_model = create_model_copy(weight_loaded_model) #recompiles the model with my new metric
        weight_loaded_model = create_model_copy(weight_loaded_model) #recompiles the model with my new metric
    
    is_split_data = False
    is_test_whole_set = False # specifies if you want to test the whole data set in one go.  Note only is_split_data or is_test_whole_set can be activated
    if is_split_data:
        # results from this are not guaranteed to match results from actual training
        print('splitting data')
        split_percentages = {"train": 0.9, "val": 0.05, "test": 0.05}
        dataset_in = util.load_data(data_file)
        dataset_processed = util.shift_data_set(dataset_in,num_obs,is_shift_data)
        split_data = util.shuffle_and_split_data(dataset_processed,args.num_obs,split_percentages)
        # split_data = util.shuffle_and_split_data(dataset,num_obs,split_percentages,shuffle_data=False)
        X_train = split_data["X_train"] 
        X_val = split_data["X_val"]   
        X_test = split_data["X_test"]  

        Y_train = split_data["Y_train"] 
        Y_val = split_data["Y_val"]   
        Y_test = split_data["Y_test"]  

        # evaluate the keras weight_loaded_model
        #TODO replace this with the function from util.py
        print('testing training data')
        train_loss, train_accuracy = weight_loaded_model.evaluate(X_train,Y_train)
        print('testing validation data')
        val_loss, val_accuracy = weight_loaded_model.evaluate(X_val, Y_val)
        print('testing test data')
        test_loss, test_accuracy = weight_loaded_model.evaluate(X_test, Y_test)
        print('\nTrain_Accuracy: %.2f' % (train_accuracy*100))
        print('Validation_Accuracy: %.2f' % (val_accuracy*100))
        print('Test_Accuracy: %.2f' % (test_accuracy*100))
        print('\nTrain_Loss: %.6f' % (train_loss))
        print('Validation_Loss: %.6f' % (val_loss))
        print('Test_Loss: %.6f' % (test_loss))
    elif is_test_whole_set:
        print('testing whole data set without splitting')
        split_percentages = {"train": 0.9, "val": 0.05, "test": 0.05}
        dataset_in = util.load_data(data_file)
        dataset_processed = util.shift_data_set(dataset_in,num_obs,is_shift_data)

        X, Y = util.separate_features_labels(dataset_processed,num_obs)

        print('testing test data')
        test_loss, test_bin_acc, test_sample_acc = weight_loaded_model.evaluate(X,Y, batch_size = batch)
        print(f'Sample Test Accuracy: {test_sample_acc*100:.4f}')
        print(f'Binary Test Accuracy: {test_bin_acc*100:.4f}')
        print(f'Test Loss: {test_loss:.6f}')
    else:
        print('splitting TV and test data, and testing both sets')
        split_percentages = {"train": 0.9, "val": 0.05, "test": 0.05}
        dataset_in = util.load_data(data_file)
        dataset_processed = util.shift_data_set(dataset_in,num_obs,is_shift_data)
        split_data = util.shuffle_and_split_data(dataset_processed,args.num_obs,split_percentages)
        
        X_tv, Y_tv = util.combine_test_val_data(split_data)

        X_test = split_data["X_test"]
        Y_test = split_data["Y_test"]

        print('testing tv data')
        tv_loss, tv_bin_acc, tv_sample_acc = weight_loaded_model.evaluate(X_tv,Y_tv,batch_size=batch)
        print('testing test data')
        test_loss, test_bin_acc, test_sample_acc = weight_loaded_model.evaluate(X_test,Y_test,batch_size=batch)

        print(f'\nTrain/Val Sample Accuracy: {tv_sample_acc*100:.4f}')
        print(f'Train/Val Binary Accuracy: {tv_bin_acc*100:.2f}')
        print(f'Train/Val Loss: {tv_loss:.6f}')

        print(f'\nTest Sample Accuracy: {test_sample_acc*100:.4f}')
        print(f'Test Binary Accuracy: {test_bin_acc*100:.2f}')
        print(f'Test Loss: {test_loss:.6f}')

#TODO move these functions to util_keras.py
def load_model(model_path):
    # makes model load work if results directory or keras model directory is inputted
    if model_path.endswith("keras_model"):
        dir_path = os.path.dirname(model_path)
    else:
        dir_path = model_path
        model_path = os.path.join(model_path,"keras_model")

    if os.name == 'nt':
        dir_path = dir_path.replace('/','\\')

    loaded_model = K.models.load_model(model_path)
    return loaded_model, dir_path

def load_model_with_checkpoint(model_path,epoch):
    loaded_model, dir_path = load_model(model_path)
    checkpoint_template = util.make_checkpoint_template()
    checkpoint_fname = checkpoint_template.format(epoch=epoch)
    checkpoint_fpath = os.path.join(dir_path,'weight_checkpoints',checkpoint_fname)
    loaded_model.load_weights(checkpoint_fpath)
    return loaded_model

def create_model_copy(pretrained_model):
    '''Creates a copy of pretrained_model with same inputs, outputs, and weights, but compiles with new accuracy metric.'''
    learning_rate = 0.0001 #TODO add these as inputs to the function
    optimizer2use = 0 # 0 corresponds to adam, 1 for RMSprop, and 2 for SGD
    optimizer = util_keras.optimizer_selector(optimizer2use,learning_rate)
    new_model = K.models.Model(inputs=pretrained_model.input, outputs=pretrained_model.output)
    new_model.set_weights(pretrained_model.get_weights())
    new_model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['binary_accuracy',util_keras.AllOutputsCorrect()])
    return new_model

def arg_parse():
    parser = ArgumentParser(description="Script allows you to load and test a model with weights from earlier epoch.",formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", "--num_obs", type=int, default = 3, help="Specify number of obstacles in selected data set")
    parser.add_argument("-d", "--data_path", type=str, default = "./ML_code/Data/small_main_data_file_courses3.csv", help = 'Path to dataset you want to evaluate NN model on.')
    parser.add_argument("-f","--model_path", type=str, default = "./main_train_results/small_main_data_file_courses3/model_15", help = 'Path to model results you wish to pull weights from.')
    parser.add_argument("-e","--epoch", type=int, default=None, help="Chooses the weights from a given epoch to be loaded into the model.  If no epoch given, loads model weights from last epoch.")
    parser.add_argument("-b","--batch_size", type=int, default=64, help="Specify the batch size used during training to make accuracy calculation match results found during training.")
    parser.add_argument("-s","--shift", action='store_false', help="Turns off shifting dataset over by half of size of obstacle field.  Expected field size is 30mx30m so shift is -15 to each x,y coordinate")

    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    #TODO until I implement recording which samples were used for each training, the only way to make results repeatable is by training with data shuffle off
    
    #TODO: update so args used are the ones defined in util
    args = arg_parse()
    model_path = args.model_path
    data_file = args.data_path
    num_obs = args.num_obs
    epoch = args.epoch
    batch = args.batch_size
    is_shift_data = args.shift

    main(model_path,epoch,data_file,num_obs,batch,is_shift_data)
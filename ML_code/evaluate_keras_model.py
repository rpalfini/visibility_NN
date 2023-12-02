import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tensorflow import keras as K
import util
import util_keras


def main(model_path,epoch,data_file,num_obs,batch,is_shift_data,scale_value=1,is_scale_data=False,num_to_test=None):
    # This function lets us load a model with specific weights and evaluate on a new dataset
    if epoch is None:
        weight_loaded_model, _ = load_model(model_path)
    else:
        weight_loaded_model = load_model_with_checkpoint(model_path,epoch)

    is_retroactive_test = True  # Adding this option to specify if we are testing custom metric on previously trained model where we need to recompile with the new metric.  Any models created prior to 11/18 at 6:30pm require retroactive tests.

    if is_retroactive_test:
        # new_model = create_model_copy(weight_loaded_model) #recompiles the model with my new metric
        if num_to_test is None:
            weight_loaded_model = create_model_copy_with_all_output_metric(weight_loaded_model) #recompiles the model with my new metric
        else:
            weight_loaded_model = create_model_copy_with_all_n_output_metric(weight_loaded_model,num_to_test) #recompiles the model with metric testing n outputs

    
    is_split_data = False
    is_test_whole_set = True # specifies if you want to test the whole data set in one go.  Note only is_split_data or is_test_whole_set can be activated
    if is_split_data:
        # results from this are not guaranteed to match results from actual training
        print('splitting data')
        split_percentages = {"train": 0.9, "val": 0.05, "test": 0.05}
        dataset_in = util.load_data(data_file)
        dataset_processed = util.shift_data_set(dataset_in,num_obs,is_shift_data)
        dataset_processed = util.scale_data_set(dataset_processed,num_obs,scale_value,is_scale_data)
        split_data = util.shuffle_and_split_data(dataset_processed,num_obs,split_percentages)
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
        train_loss, train_accuracy = weight_loaded_model.evaluate(X_train,Y_train,batch_size = batch)
        print('testing validation data')
        val_loss, val_accuracy = weight_loaded_model.evaluate(X_val, Y_val,batch_size = batch)
        print('testing test data')
        test_loss, test_accuracy = weight_loaded_model.evaluate(X_test, Y_test,batch_size = batch)
        print('\nTrain_Accuracy: %.2f' % (train_accuracy*100))
        print('Validation_Accuracy: %.2f' % (val_accuracy*100))
        print('Test_Accuracy: %.2f' % (test_accuracy*100))
        print('\nTrain_Loss: %.6f' % (train_loss))
        print('Validation_Loss: %.6f' % (val_loss))
        print('Test_Loss: %.6f' % (test_loss))
    elif is_test_whole_set:
        print('testing whole data set without splitting')
        dataset_in = util.load_data(data_file)
        dataset_processed = util.shift_data_set(dataset_in,num_obs,is_shift_data)
        dataset_processed = util.scale_data_set(dataset_processed,num_obs,scale_value,is_scale_data)


        X, Y = util.separate_features_labels(dataset_processed,num_obs)
        print('testing test data')
        if num_to_test is None:
            test_loss, test_bin_acc, test_sample_acc = weight_loaded_model.evaluate(X,Y, batch_size = batch)
        else:
            test_loss, test_bin_acc, test_sample_acc, test_n_sample_acc = weight_loaded_model.evaluate(X,Y, batch_size = batch)
        print(f'Sample Test Accuracy: {test_sample_acc*100:.4f}')
        print(f'Binary Test Accuracy: {test_bin_acc*100:.4f}')
        print(f'Test Loss: {test_loss:.6f}')
        if num_to_test is not None:
            print(f'Sample 1-n Test Accuracy: {test_n_sample_acc*100:.4f}')
            output_val_dict = util.make_output_val_dict(test_loss=test_loss, test_bin_acc=test_bin_acc, test_sample_acc=test_sample_acc, test_n_sample_acc=test_n_sample_acc)
        return output_val_dict #currently only this one uses the output_val_dict when called outside of this script, so I will only add to this until needed in the other cases
    else:
        print('splitting TV and test data, and testing both sets')
        split_percentages = {"train": 0.9, "val": 0.05, "test": 0.05}
        dataset_in = util.load_data(data_file)
        dataset_processed = util.shift_data_set(dataset_in,num_obs,is_shift_data)
        dataset_processed = util.scale_data_set(dataset_processed,num_obs,scale_value,is_scale_data)
        split_data = util.shuffle_and_split_data(dataset_processed,num_obs,split_percentages)
        
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
        write_results_file(util.fix_path_separator(model_path),tv_loss,tv_bin_acc,tv_sample_acc,test_loss,test_bin_acc,test_sample_acc)

def write_results_file(model_path,tv_loss,tv_bin_acc,tv_sample_acc,test_loss,test_bin_acc,test_sample_acc):
    out_file = make_tv_test_file_name(model_path,tv_sample_acc,test_sample_acc)
    with open(out_file,"w") as f:
        f.write('tv_sample_acc,test_sample_acc,tv_bin_acc,test_bin_acc,tv_loss,test_loss\n')
        f.write(f'{tv_sample_acc},{test_sample_acc},{tv_bin_acc},{test_bin_acc},{tv_loss:.6f},{test_loss:.6f}\n')
        

def make_tv_test_file_name(output_dir,tv_acc,test_acc):
    ''' similar to make_results_file_name in util.py but only being used for the case of retroactive testing'''
    model_number = os.path.basename(output_dir)
    tv_acc_str = "{:.2f}".format(tv_acc)
    test_acc_str = "{:.2f}".format(test_acc)
    fname = os.path.join(output_dir,f"{model_number}_results_{tv_acc_str}_{test_acc_str}.txt")
    return fname

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
        model_path = model_path.replace('/','\\')
    
    # if is_custom_metrics:
    #     # Define a dictionary with the custom objects
    #     custom_metrics = {'AllOutputsCorrect': util_keras.AllOutputsCorrect}
    #     loaded_model = K.models.load_model(model_path,custom_objects=custom_metrics)
    # else:

    #TODO: this should be implemented differently, but running out of time on project
    try:
        print(f'Loading model at {model_path}')
        loaded_model = K.models.load_model(model_path)
    except ValueError as e:
        if "Unable to restore custom object of type" in str(e):
            # Handle the specific ValueError related to custom objects
            print("Trying to reload model with custom metric AllOutputsCorrect")
            custom_metrics = {'AllOutputsCorrect': util_keras.AllOutputsCorrect}
            loaded_model = K.models.load_model(model_path,custom_objects=custom_metrics)
        else:
            # If it's a different ValueError, re-raise the exception
            raise
    return loaded_model, dir_path

def load_model_with_checkpoint(model_path,epoch):
    loaded_model, dir_path = load_model(model_path)
    checkpoint_template = util.make_checkpoint_template()
    checkpoint_fname = checkpoint_template.format(epoch=epoch)
    checkpoint_fpath = os.path.join(dir_path,'weight_checkpoints',checkpoint_fname)
    loaded_model.load_weights(checkpoint_fpath)
    return loaded_model

def create_model_copy_with_all_output_metric(pretrained_model):
    '''Creates a copy of pretrained_model with same inputs, outputs, and weights, but compiles with new accuracy metric.'''
    learning_rate = 0.0001 #TODO add these as inputs to the function
    optimizer2use = 0 # 0 corresponds to adam, 1 for RMSprop, and 2 for SGD
    optimizer = util_keras.optimizer_selector(optimizer2use,learning_rate)
    new_model = K.models.Model(inputs=pretrained_model.input, outputs=pretrained_model.output)
    new_model.set_weights(pretrained_model.get_weights())
    new_model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['binary_accuracy',util_keras.AllOutputsCorrect()])
    # if debugging, use run_eagerly=True
    # new_model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['binary_accuracy',util_keras.AllOutputsCorrect(),run_eagerly=True]) 
    return new_model

def create_model_copy_with_all_n_output_metric(pretrained_model,num2test):
    '''Creates a copy of pretrained_model with same inputs, outputs, and weights, but compiles with new accuracy metrics.'''
    learning_rate = 0.0001 #TODO add these as inputs to the function
    optimizer2use = 0 # 0 corresponds to adam, 1 for RMSprop, and 2 for SGD
    optimizer = util_keras.optimizer_selector(optimizer2use,learning_rate)
    new_model = K.models.Model(inputs=pretrained_model.input, outputs=pretrained_model.output)
    new_model.set_weights(pretrained_model.get_weights())
    new_model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['binary_accuracy',util_keras.AllOutputsCorrect(),util_keras.All_n_OutputsCorrect(n=num2test)])
    # if debugging, use run_eagerly=True
    # new_model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['binary_accuracy',util_keras.All_n_OutputsCorrect(n=num2test)],run_eagerly=True)
    # new_model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['binary_accuracy',util_keras.AllOutputsCorrect(),util_keras.All_n_OutputsCorrect(n=num2test)],run_eagerly=True)
    return new_model

def arg_parse():
    parser = ArgumentParser(description="Script allows you to load and test a model with weights from earlier epoch.",formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", "--num_obs", type=int, default = 3, help="Specify number of obstacles in selected data set")
    parser.add_argument("-d", "--data_path", type=str, default = "./ML_code/Data/small_main_data_file_courses3.csv", help = 'Path to dataset you want to evaluate NN model on.')
    parser.add_argument("-f","--model_path", type=str, default = "./old_main_train_results/small_main_data_file_courses3/model_15", help = 'Path to model results you wish to pull weights from.')
    parser.add_argument("-e","--epoch", type=int, default=None, help="Chooses the weights from a given epoch to be loaded into the model.  If no epoch given, loads model weights from last epoch.")
    parser.add_argument("-b","--batch_size", type=int, default=64, help="Specify the batch size used during training to make accuracy calculation match results found during training.")
    parser.add_argument("-s","--shift", action='store_false', help="Turns off shifting dataset over by half of size of obstacle field.  Expected field size is 30mx30m so shift is -15 to each x,y coordinate")
    parser.add_argument("-sf","--scale_flag", action='store_true', help="Turns on scaling inputs based on the argument scale_value.  This scales all of the features by the scale_value.")
    parser.add_argument("-sv","--scale_value", type=float, default=1, help="Scale value used when scale_flag is activated.  it is what the data is divided by. So 30 results in 1/30 scale")
    parser.add_argument("-nt","--num_to_test", type=int, default=None, help="used when trying to test a model with N outputs on a dataset that has less than N outputs. This value specifies the number of outputs our metric should test.")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    #TODO until I implement recording which samples were used for each training, the only way to make results repeatable is by training with data shuffle off

    args = arg_parse()
    model_path = args.model_path
    data_file = args.data_path
    num_obs = args.num_obs
    epoch = args.epoch
    batch = args.batch_size
    is_shift_data = args.shift
    num_to_test = args.num_to_test
    scale_val = args.scale_value
    scale_flag = args.scale_flag



    # main(model_path,epoch,data_file,num_obs,batch,is_shift_data,num_to_test=num_to_test)
    main(model_path,epoch,data_file,num_obs,batch,is_shift_data,num_to_test=num_to_test,scale_value=scale_val,is_scale_data=scale_flag)
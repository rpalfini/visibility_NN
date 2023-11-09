import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tensorflow import keras as K
import util


def main(model_path,epoch,data_file,num_obs):
    # This function lets us load a model with specific weights and evaluate on a new dataset
    weight_loaded_model = load_model_with_checkpoint(model_path,epoch)
    
    
    is_shift_data = True # this shifts the data over so that the course is center around origin of (0,0)
    is_split_data = False
    if is_split_data:
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
        print('testing training data')
        _, train_accuracy = weight_loaded_model.evaluate(X_train, Y_train)
        print('testing validation data')
        _, val_accuracy = weight_loaded_model.evaluate(X_val, Y_val)
        print('testing test data')
        _, test_accuracy = weight_loaded_model.evaluate(X_test, Y_test)
        print('Train_Accuracy: %.2f' % (train_accuracy*100))
        print('Validation_Accuracy: %.2f' % (val_accuracy*100))
        print('Test_Accuracy: %.2f' % (test_accuracy*100))
    else:
        print('not splitting data')
        dataset_in = util.load_data(data_file)
        dataset_processed = util.shift_data_set(dataset_in,num_obs,is_shift_data)
        X, Y = util.separate_features_labels(dataset_processed,num_obs)
        print('testing all data')
        loss, accuracy = weight_loaded_model.evaluate(X,Y)
        print('Accuracy: %.2f' % (accuracy*100))
        print(f'Loss: {loss}')


def load_model_with_checkpoint(model_path,epoch):
    # makes model load work if results directory or keras model directory is inputted
    if model_path.endswith("keras_model"):
        dir_path = os.path.dirname(model_path)
    else:
        dir_path = model_path
        model_path = os.path.join(model_path,"keras_model")

    if os.name == 'nt':
        dir_path = dir_path.replace('/','\\')
        print(dir_path)

    loaded_model = K.models.load_model(model_path)
    checkpoint_template = util.make_checkpoint_template()
    checkpoint_fname = checkpoint_template.format(epoch=epoch)
    checkpoint_fpath = os.path.join(dir_path,'weight_checkpoints',checkpoint_fname)
    # checkpoint_model = K.models.load_model(checkpoint_fpath)
    loaded_model.load_weights(checkpoint_fpath)
    # loaded_model.set_weights(checkpoint_model.get_weights())
    return loaded_model

def arg_parse():
    parser = ArgumentParser(description="Script allows you to load and test a model with weights from earlier epoch.",formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", "--num_obs", type=int, default = 3, help="Specify number of obstacles in selected data set")
    parser.add_argument("-d", "--data_path", type=str, default = "./ML_code/Data/small_main_data_file_courses3.csv", help = 'Path to dataset you want to evaluate NN model on.')
    parser.add_argument("-f","--model_path", type=str, default = "./main_train_results/small_main_data_file_courses3/model_15", help = 'Path to model results you wish to pull weights from.')
    parser.add_argument("-e","--epoch", type=int, default=2, help="Chooses the weights from a given epoch to be loaded into the model")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    #TODO until I implement recording which samples were used for each training, the only way to make results repeatable is by training with data shuffle off
    args = arg_parse()
    model_path = args.model_path
    data_file = args.data_path
    num_obs = args.num_obs
    epoch = args.epoch

    main(model_path,epoch,data_file,num_obs)
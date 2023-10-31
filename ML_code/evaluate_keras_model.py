import os
from tensorflow import keras as K
import util


def main(model_path,epoch,data_file,num_obs):
    # This function lets us load a model with specific weights and evaluate on a new dataset
    weight_loaded_model = load_model_with_checkpoint(model_path,epoch)
    
    split_percentages = [0.9, 0.05, 0.05]
    dataset = util.load_data(data_file)
    split_data = util.shuffle_and_split_data(dataset,num_obs,split_percentages,shuffle_data=False)
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
    

def load_model_with_checkpoint(model_path,epoch):
    # makes model load work if results directory or keras model directory is inputted
    if model_path.endswith("keras_model"):
        dir_path = os.path.dirname(model_path)
    else:
        dir_path = model_path
        model_path = os.path.join(model_path,"keras_model")

    if os.name == 'nt':
        dir_path = dir_path.replace('/','\\')

    loaded_model = K.models.load_model(model_path)
    checkpoint_template = util.make_checkpoint_template()
    checkpoint_fname = checkpoint_template.format(epoch=epoch)
    checkpoint_fpath = os.path.join(dir_path,'weight_checkpoints',checkpoint_fname)
    # checkpoint_model = K.models.load_model(checkpoint_fpath)
    loaded_model.load_weights(checkpoint_fpath)
    # loaded_model.set_weights(checkpoint_model.get_weights())
    return loaded_model
    


if __name__ == "__main__":
    model_path = 'C:/Users/Robert/git/visibility_NN/main_train_results/small_main_data_file_courses3/model_15/keras_model'
    # model_path = 'C:/Users/Robert/git/visibility_NN/main_train_results/small_main_data_file_courses3/model_10'
    data_file = 'C:/Users/Robert/git/visibility_NN/ML_code/Data/small_main_data_file_courses3.csv'
    num_obs = 3

    epoch = 3
    main(model_path,epoch,data_file,num_obs)
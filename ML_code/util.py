import os
import pickle

# place to store functions in project

def get_dir_list(path):
    result = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path,name))]
    return result

def init_data_store_folder(data_file):
    data_path = ('./'+data_file)
    dir_exists = os.path.isdir(data_path)
    if dir_exists:
        model_dirs = get_dir_list(data_path)
        model_folder = f'/model_{len(model_dirs)+1}'
        data_store_folder = data_path+model_folder
        os.mkdir(data_store_folder)
    else:
        os.mkdir(data_path)
        model_folder = '/model_1'
        data_store_folder = data_path+model_folder
        os.mkdir(data_store_folder)
    return data_store_folder

def record_model_results(output_dir,epochs, batch_size, train_acc, val_acc):
    f = open(output_dir+"/results.txt","w")
    f.write('epochs,batch_size,train_acc,val_acc\n')
    f.write(f'{epochs},{batch_size},{train_acc},{val_acc}\n')
    f.close()

def record_model_fit_results(results, output_folder):
    model_number,model_results_path = split_fname_path(output_folder)
    fname = f'/{model_number}_results.pkl'
    PK_fname = output_folder + fname
    Temp = open(PK_fname,'wb')
    pickle.dump(results.history,Temp)
    Temp.close()

def split_fname_path(data_path):
    '''splits a file name from its path and returns both'''
    tokens = data_path.split('/')
    fname = tokens[-1]
    fpath = "/".join(tokens[:-1])
    fpath += "/"
    return fname,fpath
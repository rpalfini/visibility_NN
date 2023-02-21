import os

# place to store functions in project

def get_dir_list(path):
    result = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path,name))]
    return result

def init_data_store_folder(data_file):
    data_path = ('./'+data_file)
    dir_exists = os.path.isdir(data_path)
    if dir_exists:
        model_dirs = get_dir_list(data_path)
        data_store_folder = f'./model_{len(model_dirs)+1}'
        os.mkdir(data_path+data_store_folder)
    else:
        os.mkdir(data_path)
        data_store_folder = './model_1'
        os.mkdir(data_path+data_store_folder)

    return data_store_folder
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tensorflow import keras as K
import re
import util
import util_keras
import evaluate_keras_model as ekm

# Using this script to calculate tv and test accuracy on all my old models that i calcualted iwth the wrong accuracy value
def extract_info(file_path):
    # Check if the file name matches the specified pattern
    if re.match(r'results\d+\.txt', os.path.basename(file_path)):
        with open(file_path, 'r') as file:
            content = file.read()

            # Extract file name
            file_name_match = re.search(r'trained on file (.+)', content)
            file_name = file_name_match.group(1) if file_name_match else 'Default_File_Name'
            print(f"File Name: {file_name}")

            # Extract is data shift value
            shift_match = re.search(r'is data shift so course is centered on origin = (\w+)', content)
            shift_value = shift_match.group(1) if shift_match else 'Default_Shift_Value'
            print(f"Is Data Shift: {shift_value}")

            # Extract scaling factor
            scaling_match = re.search(r'data is scaled by (\d+\.\d+)', content)
            scaling_factor = float(scaling_match.group(1)) if scaling_match else 1.0
            print(f"Scaling Factor: {scaling_factor}")

# # Specify the directory where the files are located
# directory_path = '/path/to/your/directory'

# # List all files in the directory
# files = os.listdir(directory_path)

# # Iterate through each file and extract information
# for file in files:
#     file_path = os.path.join(directory_path, file)
#     extract_info(file_path)
#     print('-' * 50)  # Separate output for each file


if __name__ == "__main__":
    
    args = ekm.arg_parse()

    results_folder = "./old_main_train_results"



    ekm.main(model_path,epoch,data_file,num_obs,batch,is_shift_data)

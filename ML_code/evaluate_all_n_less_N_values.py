import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tensorflow import keras as K
import datetime
import util
import util_keras
import evaluate_keras_model as ekm

'''This file is used to evaluate a given model on all the datasets with less obstacles than the model'''

class result_record:
    def __init__(self,model_tested_path,data_tested_dir):
        self.record = {}
        self.model_tested_path = model_tested_path
        self.data_tested_dir = data_tested_dir
        self.output_fname = self.make_output_fname()

    def make_output_fname(self):
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y_%m_%d_%H%M")
        file_name = f"[{formatted_time}]less_than_20_results.txt"
        return file_name

    def save_result(self,model_num,result_dict):
        self.record[model_num] = result_dict

    def output2file(self):
        '''writes results to file'''
        with open(self.output_fname,'w') as f:
            f.write(f'model tested path = {self.model_tested_path}\n')
            f.write(f'data tested directory = {self.data_tested_dir}\n')
            for model_num, test_vals in self.record.items():
                f.write(f"{model_num}: {test_vals}\n")
            

def main(args):

    model_path = args.model_path
    # data_file = args.data_path
    num_obs = args.num_obs
    epoch = args.epoch
    batch = args.batch_size
    # num_to_test = args.num_to_test
    # we require for this test that these values match what is specified below,
    # scale_val = args.scale_value
    # scale_flag = args.scale_flag
    # is_shift_data = args.shift
    scale_val = 1
    scale_flag = False
    is_shift_data = False

    # file_extension = "csv"
    file_extension = "npy"
    # base_data_path = "D:/Vis_network_data/data_file_by_course_padded/main_data_file_courses"
    base_data_path = "D:/Vis_data/data_file_by_course_transformations/shift_padded_courses_x_to_15/main_data_f_networkile_courses"
    results_record = result_record(model_path,base_data_path)
    
    # base_data_path = util.fix_path_separator(base_data_path)
    # model = 13
    model_range = range(args.model_range[0],args.model_range[1]+1)

    
    try:
        for model in model_range: # we don't need to test on the data set that has the expected number of obstacles
            num_to_test = model
            data_file = f"{base_data_path}{model}.{file_extension}"
            results_dict = ekm.main(model_path,epoch, data_file,num_obs,batch,is_shift_data,num_to_test=num_to_test,scale_value=scale_val,is_scale_data=scale_flag)
            results_record.save_result(model,results_dict)
        
        # num_to_test = model
        # data_file = f"{base_data_path}{model}.csv"
        # results_dict = ekm.main(model_path,epoch, data_file,num_obs,batch,is_shift_data,num_to_test=num_to_test,scale_value=scale_val,is_scale_data=scale_flag)
        # results_record.save_result(model,results_dict)
    except:
        results_record.output2file()    

    results_record.output2file()

    

if __name__ == "__main__":
    args = ekm.arg_parse()
    
    main(args)



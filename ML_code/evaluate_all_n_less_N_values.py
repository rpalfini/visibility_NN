import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tensorflow import keras as K
import datetime
import util
import util_keras
import evaluate_keras_model as ekm

'''This file is used to evaluate a given model on all the datasets with less obstacles than the model'''

class result_record:
    def __init__(self,model_tested_path):
        self.record = {}
        self.model_tested_path = model_tested_path
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
            for model_num, test_vals in self.record.items():
                f.write(f"{model_num}: {test_vals}\n")
            

def main(args):

    model_path = args.model_path
    # data_file = args.data_path
    num_obs = args.num_obs
    epoch = args.epoch
    batch = args.batch_size
    num_to_test = args.num_to_test
    # we require for this test that these values match what is specified below,
    # scale_val = args.scale_value
    # scale_flag = args.scale_flag
    # is_shift_data = args.shift
    scale_val = 1
    scale_flag = False
    is_shift_data = False

    results_record = result_record(model_path)

    base_data_path = "D:/Vis_network_data/data_file_by_course_padded/main_data_file_courses"
    base_data_path = util.fix_path_separator(base_data_path)

    for model in range(1,20): # we don't need to test on the data set that has the expected number of obstacles
        data_file = f"{base_data_path}_{model}.csv"
        results_dict = ekm.main(model_path,epoch, data_file,num_obs,batch,is_shift_data,num_to_test=num_to_test,scale_value=scale_val,is_scale_data=scale_flag)
        results_record = results_record.save_result(model,results_dict)

    results_record.output2file()

    

if __name__ == "__main__":
    args = ekm.arg_parse()
    
    main(args)



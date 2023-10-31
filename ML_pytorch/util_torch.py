import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import config
from ML_code import util

# # Define a custom dataset class
class CustomDataset(Dataset):
    '''Class is used to load dataset from csv file'''
    def __init__(self, csv_file, feat_size, label_size, transform=None):
        self.data = pd.read_csv(csv_file)
        self.feat_size = feat_size
        self.label_size = label_size
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data.iloc[idx,:self.feat_size].values.astype(float)
        labels = self.data.iloc[idx,self.feat_size:-1].values.astype(float) #last column is the cost of the optimal path as found during data generation
        
        # verify labels are the correct length
        if labels.shape[0] != self.label_size:
            raise Exception('incorrect number of labels')

        if self.transform:
            features = self.transform(features)

        features = torch.Tensor(features)
        labels = torch.Tensor(labels)

        return features, labels

# class CustomDataset(Dataset):
#     def __init__(self, data_path, num_feat,num_label):
#         # Load data from NumPy files
#         self.data = util.load_data(data_path)
#         self.feat_size = num_feat
#         self.label_size = num_label
#         self.transform = None  # You can add data transformations here

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         # Retrieve a single data sample and its corresponding label
#         features = self.data[idx,:self.feat_size]
#         labels = self.data[idx,self.feat_size:-1] #last column is the cost of the optimal path as found during data generation

#         if labels.shape[0] != self.label_size:
#             raise Exception('incorrect number of labels')

#         if self.transform:
#             features = self.transform(features)

#         return features, labels


# def example_seq_model():
#     model = nn.Sequential(
#         nn.Linear(784, 128),  # Input size: 784, Output size: 128
#         nn.ReLU(),            # ReLU activation function
#         nn.Linear(128, 64),   # Input size: 128, Output size: 64
#         nn.ReLU(),
#         nn.Linear(64, 10),    # Input size: 64, Output size: 10 (assuming a classification task)
#     )
#     return model

def three_obs_nn(num_feat,num_labels):
    model = nn.Sequential(
        nn.Linear(num_feat,10),
        nn.ReLU(),
        nn.Linear(10,20),
        nn.ReLU(),
        nn.Linear(20,20),
        nn.ReLU(),
        nn.Linear(20,num_labels),
        nn.Sigmoid()
    )
    return model

def calc_num_feat_label(num_obstacles):
    features = 3*num_obstacles + 4
    labels = num_obstacles
    return features, labels

# def arg_parse():
#     parser = ArgumentParser(description="Pytorch Model Training.  Used for script that is training model based on data file",formatter_class=ArgumentDefaultsHelpFormatter)
#     parser.add_argument("-n", "--num_obs", type=int, default = 3, help="Specify number of obstacles in selected data set")
#     parser.add_argument("-f", "--file_path", type=str, default = "./ML_code/Data/small_main_data_file_courses3.csv")
#     parser.add_argument("-b","--batch_size", type=int, default=64, help="set batch size for training")
#     parser.add_argument("-e","--n_epochs", type=int, default=100, help="sets number of epochs for the data")
#     args = parser.parse_args()
#     return args
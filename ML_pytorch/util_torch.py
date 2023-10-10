import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data.iloc[idx, 0]  # Assuming the features are in the first column
        labels = self.data.iloc[idx, 1]  # Assuming the labels are in the second column

        if self.transform:
            features = self.transform(features)

        return features, labels

def calc_num_feat_label(num_obstacles):
    features = 3*num_obstacles + 4
    labels = num_obstacles
    return features, labels

def arg_parse():
    parser = ArgumentParser(description="Pytorch Model Training.  Used for script that is training model based on data file",formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", "--num_obs", type=int, default = 3, help="Specify number of obstacles in selected data set")
    parser.add_argument("-f", "--file_path", type=str, default = "./ML_code/Data/main_data_file_courses3.csv")
    parser.add_argument("-b","--batch_size", type=int, default=64, help="set batch size for training")
    parser.add_argument("-e","--n_epochs", type=int, default=100, help="sets number of epochs for the data")
    args = parser.parse_args()
    return args
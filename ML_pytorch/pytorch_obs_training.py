import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# importing custom modules
import config
import util_torch
from ML_code import util

# converting keras code to pytorch to leverage use of DataLoader class
def main():

    args = util.arg_parse()

    data_file = args.file_path
    num_feat, num_label = util_torch.calc_num_feat_label(args.num_obs)

    # Create a custom dataset and data loader
    dataset = util_torch.CustomDataset(args.file_path,num_feat,num_label)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = util_torch.three_obs_nn(num_feat,num_label)

    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) #specifying lr is optional, the default is 0.001

    # Training loop
    num_epochs = args.n_epochs
    print(f'num epochs is {num_epochs}')
    for epoch in range(num_epochs):
        for inputs, labels in data_loader:
            # Forward pass
            outputs = model(inputs)
            
            # Compute the loss
            loss = criterion(outputs, labels)
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print('Training finished!')

if __name__ == "__main__":
    main()
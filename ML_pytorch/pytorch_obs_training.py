import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import util_torch as util

def main():

    args = util.arg_parse()

    data_file = args.file_path
    features, labels = util.calc_num_feat_label(args.num_obstacles)

    # Create a custom dataset and data loader
    dataset = CustomDataset(train_data, train_labels)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Define a simple Sequential model
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 10
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
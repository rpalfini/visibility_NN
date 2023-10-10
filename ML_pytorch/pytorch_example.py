# example code using pytorch for ML instead of keras

import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple Sequential model
model = nn.Sequential(
    nn.Linear(784, 128),  # Input size: 784, Output size: 128
    nn.ReLU(),            # ReLU activation function
    nn.Linear(128, 64),   # Input size: 128, Output size: 64
    nn.ReLU(),
    nn.Linear(64, 10),    # Input size: 64, Output size: 10 (assuming a classification task)
)

# Define a loss function (e.g., CrossEntropyLoss for classification)
criterion = nn.CrossEntropyLoss()

# Define an optimizer (e.g., SGD)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy data (replace with your dataset)
input_data = torch.randn(64, 784)  # Batch size: 64, Input size: 784
labels = torch.randint(0, 10, (64,))  # Batch size: 64, Number of classes: 10

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(input_data)
    
    # Compute the loss
    loss = criterion(outputs, labels)
    
    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print('Training finished!')

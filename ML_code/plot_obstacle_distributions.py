import numpy as np
import matplotlib.pyplot as plt

import util

# Load CSV file into a numpy array
# filename = '20_obs_row_changed.txt'
filename = 'new_data_20_obs_row_changed.txt'
data = np.genfromtxt(filename, delimiter=',')
print(f'data.shape = {data.shape}')
# Select columns 5 to 64
selected_columns = data[:, 4:64]
# selected_columns = data[142:, 4:64]
print(f'selected_columns.shape = {selected_columns.shape}')
# Plot histograms for each column
num_columns = selected_columns.shape[1]
num_rows = num_columns // 3 + (num_columns % 3 > 0)  # 3 histograms per row

fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows),sharex='col')

column_names = ['X Center', 'Y Center', 'Radius']

for i in range(num_columns):
    row = i // 3
    col = i % 3
    axes[row, col].hist(selected_columns[:, i], bins=20, edgecolor='black')
    if i < 3:
        axes[row, col].set_title(column_names[i])

# Plot histograms for the first three columns in a separate plot
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))


for i in range(3):
    axes2[i].hist(selected_columns[:, i], bins=20, edgecolor='black')
    axes2[i].set_title(column_names[i])

# Adjust layout
plt.tight_layout()
plt.show()


# # Adjust layout to prevent overlapping titles
# plt.tight_layout()
# plt.show()

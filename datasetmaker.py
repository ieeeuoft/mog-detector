import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
# Define the custom dataset
class CSVDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.X = self.data.iloc[:, :-1].values
        self.y = self.data.iloc[:, -1].values
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return sample, label

def scale_coordinates(input_csv, output_csv):
    # Read the CSV file with header
    data = pd.read_csv(input_csv)
    
    # Ensure the data has the correct number of columns (25 including the label)
    processed_data = []
    # Process each sample
    for idx in range(len(data)):
        # Extract coordinates and label
        coordinates = data.iloc[idx, :-1].values  # First 24 columns are coordinates
        label = data.iloc[idx, -1]  # Last column is the label

        # Extract x, y, and z coordinates
        x_coords = coordinates[::3]
        y_coords = coordinates[1::3]
        z_coords = coordinates[2::3]
        
        # Scale x-coordinates between 0 and 1 based on the sample-specific max and min
        x_min = x_coords.min()
        x_max = x_coords.max()
        if x_max != x_min:  # Avoid division by zero
            scaled_x_coords = (x_coords - x_min) / (x_max - x_min)
        else:
            scaled_x_coords = x_coords  # If all x are the same, no scaling needed
        
        # Scale y-coordinates between 0 and 1 based on the sample-specific max and min
        y_min = y_coords.min()
        y_max = y_coords.max()
        if y_max != y_min:  # Avoid division by zero
            scaled_y_coords = (y_coords - y_min) / (y_max - y_min)
        else:
            scaled_y_coords = y_coords  # If all y are the same, no scaling needed
        
        # Scale z-coordinates between 0 and 1 based on the sample-specific max and min
        z_min = z_coords.min()
        z_max = z_coords.max()
        if z_max != z_min:  # Avoid division by zero
            scaled_z_coords = (z_coords - z_min) / (z_max - z_min)
        else:
            scaled_z_coords = z_coords  # If all z are the same, no scaling needed
        
        new_row = np.empty(64)
        new_row[:-1:3] = scaled_x_coords
        new_row[1::3] = scaled_y_coords
        new_row[2::3] = scaled_z_coords
        new_row[-1] = label
        # new_row.append(label)  # Keep the label unchanged
        
        # Append the new row to the processed data list
        processed_data.append(new_row)
    
    # Convert the processed data list to a DataFrame
    processed_df = pd.DataFrame(processed_data, columns=data.columns)
    
    # Write the updated data to a new CSV file with header
    processed_df.to_csv(output_csv, index=False)



# Example usage
input_csv = './data/data.csv'
output_csv = './data/data_scaled.csv'
scale_coordinates(input_csv, output_csv)

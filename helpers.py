import pandas as pd
import numpy as np
import torch

from nets import HandNet
def scale_single_coord(coordinates):
    # Read the CSV file with header    
    x_coords = coordinates[::3]
    y_coords = coordinates[1::3]
    z_coords = coordinates[2::3]
    
    # Scale x-coordinates between 0 and 1 based on the sample-specific max and min
    x_min = x_coords.min()
    x_max = x_coords.max()
    scaled_x_coords = (x_coords - x_min) / (x_max - x_min)
    
    # Scale y-coordinates between 0 and 1 based on the sample-specific max and min
    y_min = y_coords.min()
    y_max = y_coords.max()
    scaled_y_coords = (y_coords - y_min) / (y_max - y_min)

    
    # Scale z-coordinates between 0 and 1 based on the sample-specific max and min
    z_min = z_coords.min()
    z_max = z_coords.max()
    scaled_z_coords = (z_coords - z_min) / (z_max - z_min)

    
    new_row = np.empty(63)
    new_row[::3] = scaled_x_coords
    new_row[1::3] = scaled_y_coords
    new_row[2::3] = scaled_z_coords

    return new_row


def predict(model, scaled_coords: list) -> str:
   scaled_coords = np.array(scaled_coords)
   # scaled_coords = scale_single_coord(scaled_coords)
   output = model(torch.tensor(scaled_coords, dtype=torch.float32))
   labels = ['peace sign', 'euro footballer', 'thumbs up', 'kpop heart', 'what the sigma']
   print(output)
   if output[4] > 0.5:
      return 'what the sigma'
   if torch.max(output) > 2:
      return labels[torch.argmax(output)]
   else:
      return None
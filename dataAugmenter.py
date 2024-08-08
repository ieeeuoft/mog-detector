import os
import csv
import numpy as np

def augment_one(coordinates):
    # TRY 1 5 9 17
    row = coordinates.copy()
    vec1 = np.array([float(row[12]) - float(row[3]), float(row[13]) - float(row[4]), float(row[14]) - float(row[5])])
    #pinky: ID17 to ID20
    vec2 = np.array([float(row[60]) - float(row[51]), float(row[61]) - float(row[52]), float(row[62]) - float(row[53])])
    
    #we want to find the vectors for tip thumb and tip of pinky, to bring more confidence to the One Hand Heart
    #tip of thumb: ID3 to ID4
    vec3 = np.array([float(row[12]) - float(row[9]), float(row[13]) - float(row[10]), float(row[14]) - float(row[11])])
    #tip of index: ID7 to ID8
    vec4 = np.array([float(row[24]) - float(row[21]), float(row[25]) - float(row[22]), float(row[26]) - float(row[23])])
    
    #we want to find the vectors for middle finger and index finger, to bring more confidence to the Peace Sign
    #index: ID5 to ID8
    vec5 = np.array([float(row[24]) - float(row[15]), float(row[25]) - float(row[16]), float(row[26]) - float(row[17])])
    #middle finger: ID9 to ID12
    vec6 = np.array([float(row[36]) - float(row[27]), float(row[37]) - float(row[28]), float(row[38]) - float(row[29])])
    
    # Compute dot products
    val1 = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    val2 = np.dot(vec3, vec4) / (np.linalg.norm(vec3) * np.linalg.norm(vec4))
    val3 = np.dot(vec5, vec6) / (np.linalg.norm(vec5) * np.linalg.norm(vec6))
    # Add the dot products to the row
    row.append(val1)
    row.append(val2)
    row.append(val3)
    return row

def main():
  output_file = "./data/data_2_aug_1.csv"
  input_file = "./data/data_2.csv"

  if not os.path.isfile(input_file):
      raise FileNotFoundError(f"File not found: {input_file}")

  with open(input_file, mode='r') as infile, open(output_file, mode='w', newline='') as outfile:
      reader = csv.reader(infile)
      writer = csv.writer(outfile)
      
      # Read the header
      header = next(reader)
      
      # Add the new columns to the header
      label = header.pop()
      header.append('DotProd1_ThumbsUp')
      header.append('DotProd2_Heart')
      header.append('DotProd3_Peace')
      header.append(label)
      
      # Write the new header to the output file
      writer.writerow(header)
      
      # Iterate over each row to compute the dot product
      for row in reader:
          try:
              row2 = row.copy()
              #we want to find the vectors for thumb and pinky, to differentiate Calling & Thumbs Up
              #thumb: ID0 to ID4
              vec1 = np.array([float(row[12]) - float(row[3]), float(row[13]) - float(row[4]), float(row[14]) - float(row[5])])
              #pinky: ID17 to ID20
              vec2 = np.array([float(row[60]) - float(row[51]), float(row[61]) - float(row[52]), float(row[62]) - float(row[53])])
              
              #we want to find the vectors for tip thumb and tip of pinky, to bring more confidence to the One Hand Heart
              #tip of thumb: ID3 to ID4
              vec3 = np.array([float(row[12]) - float(row[9]), float(row[13]) - float(row[10]), float(row[14]) - float(row[11])])
              #tip of index: ID7 to ID8
              vec4 = np.array([float(row[24]) - float(row[21]), float(row[25]) - float(row[22]), float(row[26]) - float(row[23])])
              
              #we want to find the vectors for middle finger and index finger, to bring more confidence to the Peace Sign
              #index: ID5 to ID8
              vec5 = np.array([float(row[24]) - float(row[15]), float(row[25]) - float(row[16]), float(row[26]) - float(row[17])])
              #middle finger: ID9 to ID12
              vec6 = np.array([float(row[36]) - float(row[27]), float(row[37]) - float(row[28]), float(row[38]) - float(row[29])])
              
              # Compute dot products
              val1 = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
              val2 = np.dot(vec3, vec4) / (np.linalg.norm(vec3) * np.linalg.norm(vec4))
              val3 = np.dot(vec5, vec6) / (np.linalg.norm(vec5) * np.linalg.norm(vec6))
              # Add the dot products to the row
              label = row2.pop()
              row2.append(val1)
              row2.append(val2)
              row2.append(val3)
              row2.append(label)
              
              
              # Write the updated row to the output file
              writer.writerow(row2)
          except ValueError as e:
              print(f"Skipping row due to error: {row}, error: {e}")

  print(f"Output file saved as: {output_file}")
# main()
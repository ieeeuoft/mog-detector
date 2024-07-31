import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from datasetmaker import CSVDataset
from nets import HandNet, FaceNet, HandNet2

import os
import argparse

# CONSTANTS
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
MODEL_TYPE = None

def get_model(model_type):
  if model_type == 'hand' or model_type == 0:
    return HandNet()
  elif model_type =="hand2":
    return HandNet2()
  elif model_type == 'face' or model_type == 1:
    return FaceNet()
  else:
    raise ValueError('Invalid model type')

# Load dataset
def get_dataloaders(data_path):
  dataset = CSVDataset(data_path)
  dataset.check()
  # Split dataset into training and test sets
  train_size = int(0.8 * len(dataset))
  test_size = len(dataset) - train_size
  train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
  return train_loader, test_loader

# Define model, loss function, and optimizer
def train_model(model_type, train_loader, test_loader):
  model = get_model(model_type)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

  # Training loop
  num_epochs = NUM_EPOCHS

  for epoch in range(num_epochs):
      model.train()
      running_loss = 0.0
      for inputs, labels in train_loader:
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()

      print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f},', end=' ')
      evaluate_model(model, test_loader)
  return model

# Evaluation on the test set
def evaluate_model(model, test_loader):
  model.eval()
  correct = 0
  total = 0
  with torch.no_grad():
      for inputs, labels in test_loader:
          outputs = model(inputs)
          _, predicted = torch.max(outputs, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
  print(f'Accuracy: {100 * correct / total:.2f}%')

# Save model in /models directory
def save_model(model):
  if not os.path.exists('./models'):
    os.makedirs('./models')
  torch.save(model.state_dict(), f'./models/{MODEL_TYPE}_model_{0}_scaled_epoch{NUM_EPOCHS}_lr{LEARNING_RATE}_bs_{BATCH_SIZE}.pth')

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Train a hand or face gesture recognition model')
  parser.add_argument('model_type', type=str, help='Type of model to train: hand or face')
  parser.add_argument('data_path', type=str, help='Path to the CSV file containing the data')
  parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
  parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
  parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs for training')
  args = parser.parse_args()

  MODEL_TYPE = args.model_type
  DATA_PATH = args.data_path
  if args.batch_size: BATCH_SIZE = args.batch_size
  if args.learning_rate: LEARNING_RATE = args.learning_rate
  if args.num_epochs: NUM_EPOCHS = args.num_epochs


  train_loader, test_loader = get_dataloaders(DATA_PATH)
  print("Data loaded successfully.")
  
  model = train_model(MODEL_TYPE, train_loader, test_loader)
  evaluate_model(model, test_loader)
  print("Model training and evaluation completed successfully.")

  save_model(model)
  print("Model saved successfully.")

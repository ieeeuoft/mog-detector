import torch
import torch.nn as nn
import torch.nn.functional as F

class HandNet(nn.Module):
  def __init__(self):
    super(HandNet, self).__init__()
    self.layer1 = nn.Linear(21*3, 45)
    self.layer2 = nn.Linear(45, 35)
    self.layer3 = nn.Linear(35, 5)
  def forward(self, img):
    # flattened = img.view(-1, 21 * 3)
    activation1 = F.relu(self.layer1(img))
    activation2 = F.relu(self.layer2(activation1))
    output = self.layer3(activation2)
    return output
  def setup(self, PATH: str):
    self.load_state_dict(torch.load(PATH))
    return self
  
class HandNet2(nn.Module):
  def __init__(self):
    super(HandNet2, self).__init__()
    self.layer1 = nn.Linear(21*3 + 3, 45)
    self.layer2 = nn.Linear(45, 35)
    self.layer3 = nn.Linear(35, 5)
  def forward(self, img):
    # flattened = img.view(-1, 21 * 3)
    activation1 = F.relu(self.layer1(img))
    activation2 = F.relu(self.layer2(activation1))
    output = self.layer3(activation2)
    return output
  def setup(self, PATH: str):
    self.load_state_dict(torch.load(PATH))
    return self

class FaceNet(nn.Module):
  def __init__(self):
    super(FaceNet, self).__init__()
    self.layer1 = nn.Linear(468*3, 400)
    self.layer2 = nn.Linear(400, 100)
    self.layer3 = nn.Linear(100, 6)
  def forward(self, img):
    activation1 = F.relu(self.layer1(img))
    activation2 = F.relu(self.layer2(activation1))
    output = self.layer3(activation2)
    return output
  def setup(self, PATH: str):
    self.load_state_dict(torch.load(PATH))
    return self
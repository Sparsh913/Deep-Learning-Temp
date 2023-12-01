import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy
import matplotlib.pyplot as plt
import numpy as np
from nn import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy
import matplotlib.pyplot as plt
import numpy as np
from nn import *

train_data = scipy.io.loadmat('data/nist36_train.mat')
valid_data = scipy.io.loadmat('data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

# # Convert to torch tensors
train_x = torch.from_numpy(train_x).float()
train_y = torch.from_numpy(train_y).float()
valid_x = torch.from_numpy(valid_x).float()
valid_y = torch.from_numpy(valid_y).float()

label = np.where(train_y == 1)[1]
label = torch.from_numpy(label).long()
v_label = np.where(valid_y == 1)[1]
v_label = torch.from_numpy(v_label).long()

batch_size = 32
# Dataloader
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, label), batch_size=batch_size, shuffle=True, num_workers=4)
valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x, v_label), batch_size=batch_size, shuffle=True, num_workers=4)

# Make a Convolutional Neural Net class
class ConvNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ConvNet, self).__init__()
        # Define layers
        self.conv1 = nn.Sequential(nn.Conv2d(1, 10, 5),
                                   nn.MaxPool2d(2,2),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(10, 20, 5),
                                      nn.MaxPool2d(2,2),
                                      nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(500, 200),
                                    nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(200, 36))
        
    def forward(self, x):
        # Define forward pass
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f2 = f2.view(f2.size(0), -1)
        f3 = self.fc1(f2)
        x = self.fc2(f3)
        return x
    
# Initialize model
model = ConvNet(train_x.shape[1], 64, train_y.shape[1])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
batches = get_random_batches(train_x,train_y, batch_size)

loss = F.cross_entropy
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
train_loss = []
valid_loss = []
train_acc = []
valid_acc = []

# Train model
for epoch in range(100):
    total_loss = 0
    total_acc = 0

    for xb, yb in batches:
        # Forward
        # xb = torch.from_numpy(xb).float()
        # yb = torch.from_numpy(yb).long()
        xb = xb.to(device)
        yb = yb.to(device)
        label = np.where(yb == 1)[1]
        label = torch.from_numpy(label).long()
        output = model(xb.reshape(-1,1,32,32))
        loss = F.cross_entropy(output, yb)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Compute loss and accuracy
        total_loss += loss.item()
        total_acc += torch.sum(torch.argmax(output, dim=1) == label).item() / len(label)
    train_loss.append(total_loss / len(train_loader))
    train_acc.append(total_acc / len(train_loader))
    print("Epoch: {}, Loss: {}, Accuracy: {}".format(epoch, total_loss / len(train_loader), total_acc / len(train_loader)))
    # Validation
    with torch.no_grad():
        total_loss = 0
        total_acc = 0
        for xb, yb in valid_loader:
            # Forward
            xb = xb.to(device)
            yb = yb.to(device)
            v_label = np.where(yb == 1)[1]
            v_label = torch.from_numpy(v_label).long()
            output = model(xb)
            loss = F.cross_entropy(output, v_label)
            # Compute loss and accuracy
            total_loss += loss.item()
            total_acc += torch.sum(torch.argmax(output, dim=1) == v_label).item() / len(v_label)
        valid_loss.append(total_loss / len(valid_loader))
        valid_acc.append(total_acc / len(valid_loader))
        print("Epoch: {}, Loss: {}, Accuracy: {}".format(epoch, total_loss / len(valid_loader), total_acc / len(valid_loader)))

# Plot loss and accuracy
plt.figure()
plt.plot(train_loss, label='Training Loss')
plt.plot(valid_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')
plt.show()

plt.figure()
plt.plot(train_acc, label='Training Accuracy')
plt.plot(valid_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curve')
plt.show()


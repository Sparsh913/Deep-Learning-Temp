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
# train_x = torch.from_numpy(train_x).float()
# train_y = torch.from_numpy(train_y).float()
valid_x = torch.from_numpy(valid_x).float()
valid_y = torch.from_numpy(valid_y).float()

# Make a Neural Net class
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        # Define layers
        layers = [nn.Linear(input_size, hidden_size), nn.Sigmoid(), nn.Linear(hidden_size, output_size)]
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        # Define forward pass
        x = self.layers(x)
        return x
    
# Initialize model
model = NeuralNet(train_x.shape[1], 64, train_y.shape[1])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
batches = get_random_batches(train_x,train_y,32)

# Define loss function and optimizer
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
        xb = torch.from_numpy(xb).float()
        yb = torch.from_numpy(yb).long()
        xb = xb.to(device)
        yb = yb.to(device)
        label = np.where(yb == 1)[1]
        label = torch.tensor(label)
        probs = model(xb)
        # Loss
        l = loss(probs, label)
        # Backward
        optimizer.zero_grad()
        l.backward()
        optimizer.step()


        # Add loss and accuracy to epoch totals
        total_loss += l
        _, predicted = torch.max(probs.data, 1)

        total_acc += ((label==predicted).sum().item())

    avg_acc = total_acc / train_x.shape[0]
    avg_loss = total_loss / len(batches)
    label_v = torch.tensor(np.where(valid_y == 1)[1])
    probs_v = model(valid_x)
    l_v = loss(probs_v.data, label_v)
    _, predicted_v = torch.max(probs_v.data, 1)
    acc_v = ((label_v==predicted_v).sum().item()) / valid_x.shape[0]
    train_loss.append(avg_loss.detach().numpy())
    valid_loss.append(l_v.detach().numpy())
    train_acc.append(avg_acc)
    valid_acc.append(acc_v)

    # Print loss
    if epoch % 10 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(epoch,avg_loss,avg_acc))
print('Validation accuracy: ', acc_v)
        # print('Epoch: {}, Valid Loss: {}'.format(epoch, l_v.item()))


# Plot loss
plt.plot(train_loss)
plt.plot(valid_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
# plt.ylim(0.03, 0.095)
plt.legend(['Training Loss', 'Validation Loss'])
plt.title('Loss Curve')
plt.show()

# Plot accuracy
plt.plot(train_acc)
plt.plot(valid_acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.title('Accuracy Curve')
plt.show()
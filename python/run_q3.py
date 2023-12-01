import numpy as np
import scipy.io
from nn import *
from matplotlib import pyplot as plt

train_data = scipy.io.loadmat('data/nist36_train.mat')
valid_data = scipy.io.loadmat('data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = 32
learning_rate = 0.003
hidden_size = 64
##########################
##### your code here #####
##########################

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
##########################
##### your code here #####
##########################
n,d = train_x.shape
n, c = train_y.shape
initialize_weights(train_x.shape[1], hidden_size, params, 'layer1')
W_init = params['Wlayer1']
initialize_weights(hidden_size, c, params, 'output')

train_loss = []
train_acc = []
valid_loss = []
valid_acc = []

# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb, yb in batches:
        # training loop can be exactly the same as q2!
        ##########################
        ##### your code here #####
        ##########################
        # forward
        hidden_layer1 = forward(xb, params, 'layer1', sigmoid)
        probs = forward(hidden_layer1, params, 'output', softmax) # Probabilities
        # loss
        # be sure to add loss and accuracy to epoch totals
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        total_acc += acc

        # backward
        delta = probs - yb
        delta = backwards(delta, params, 'output', linear_deriv)
        backwards(delta, params, 'layer1', sigmoid_deriv)

        # apply gradient
        for layer in ['output', 'layer1']:
            params['W' + layer] -= learning_rate * params['grad_W' + layer]
            params['b' + layer] -= learning_rate * params['grad_b' + layer]

    # Accuracy
    avg_acc = total_acc/batch_num
    train_acc.append(avg_acc)
    # Loss
    total_loss /= n
    train_loss.append(total_loss)

        # pass

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr, total_loss, total_acc))

# run on validation set and report accuracy! should be above 75%
# valid_acc = None
##########################
##### your code here #####
##########################
    v_hidden_layer1 = forward(valid_x, params, 'layer1', sigmoid)
    v_probs = forward(v_hidden_layer1, params, 'output', softmax) # Probabilities
    v_loss, v_acc = compute_loss_and_acc(valid_y, v_probs)
    valid_loss.append(v_loss/valid_x.shape[0])
    valid_acc.append(v_acc)

    print('Validation accuracy: ',v_acc)
    if False: # view the data
        for crop in xb:
            import matplotlib.pyplot as plt
            plt.imshow(crop.reshape(32,32).T)
            plt.show()
print("Length of valid_acc: ", len(valid_acc))
plt.figure(0)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(np.arange(max_iters), train_loss, 'r')
plt.plot(np.arange(max_iters), valid_loss, 'b')
plt.legend(['training loss', 'valid loss'])
plt.figure(1)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(np.arange(max_iters), train_acc, 'r')
plt.plot(np.arange(max_iters), valid_acc, 'b')
plt.legend(['training accuracy', 'valid accuracy'])
plt.show()

import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# visualize weights here
##########################
##### your code here #####
##########################
# W1 = params['Wlayer1']
W1 = W_init
W1 = W1.T
fig = plt.figure(2)
grid = ImageGrid(fig, 111, nrows_ncols=(8, 8), axes_pad=0.1)
for i in range(64):
    grid[i].imshow(W1[i].reshape(32,32))
plt.show()

# Q3.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# compute comfusion matrix here
##########################
##### your code here #####
##########################
test_data = scipy.io.loadmat('data/nist36_test.mat')
test_x, test_y = test_data['test_data'], test_data['test_labels']
test_h1 = forward(test_x, params, 'layer1', sigmoid)
test_probs = forward(test_h1, params, 'output', softmax)
loss, acc = compute_loss_and_acc(test_y, test_probs)
print('Test accuracy: ',acc)
print('Test loss: ',loss)
for i in range(test_x.shape[0]):
    idx = np.argmax(test_probs[i, :])
    idx2 = np.argmax(test_y[i, :])
    confusion_matrix[idx2, idx] += 1    

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()
import numpy as np
import scipy.io
from nn import *
from collections import Counter

train_data = scipy.io.loadmat('data/nist36_train.mat')
valid_data = scipy.io.loadmat('data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
##########################
n,d = train_x.shape
# n, c = train_y.shape
initialize_weights(train_x.shape[1], hidden_size, params, 'layer1')
initialize_weights(hidden_size, hidden_size, params, 'layer2')
initialize_weights(hidden_size, hidden_size, params, 'layer3')
initialize_weights(hidden_size, train_x.shape[1], params, 'output')
##### your code here #####
##########################

# Momentum variables
keys = [key for key in params.keys()]
for key in keys:
    params['M_' + key] = np.zeros(params[key].shape)

train_loss = []
valid_loss = []

# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        ##########################
        ##### your code here #####
        ##########################
        
        # forward
        hidden_layer1 = forward(xb, params, 'layer1', relu)
        hidden_layer2 = forward(hidden_layer1, params, 'layer2', relu)
        hidden_layer3 = forward(hidden_layer2, params, 'layer3', relu)
        probs = forward(hidden_layer3, params, 'output', sigmoid)
        
        # loss
        loss = np.sum((xb - probs)**2)
        total_loss += loss

        # backward
        delta = probs - xb
        delta = backwards(delta, params, 'output', sigmoid_deriv)
        delta = backwards(delta, params, 'layer3', relu_deriv)
        delta = backwards(delta, params, 'layer2', relu_deriv)
        backwards(delta, params, 'layer1', relu_deriv)

        # Validation
        v_hidden_layer1 = forward(valid_x, params, 'layer1', relu)
        v_hidden_layer2 = forward(v_hidden_layer1, params, 'layer2', relu)
        v_hidden_layer3 = forward(v_hidden_layer2, params, 'layer3', relu)
        v_probs = forward(v_hidden_layer3, params, 'output', sigmoid) # Probabilities
        v_loss, _ = compute_loss_and_acc(valid_x, v_probs)
        valid_loss.append(v_loss/valid_x.shape[0])

        # apply gradient -> Weights Update 
        for layer in ['output', 'layer1', 'layer2', 'layer3']:
            params['M_W' + layer] = 0.9 * params['M_W' + layer] - learning_rate * params['grad_W' + layer]
            params['M_b' + layer] = 0.9 * params['M_b' + layer] - learning_rate * params['grad_b' + layer]
            params['W' + layer] += params['M_W' + layer]
            params['b' + layer] += params['M_b' + layer]

    train_loss.append(total_loss/n)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9
        
# Plot
import matplotlib.pyplot as plt
plt.plot(np.arange(max_iters), train_loss)
plt.plot(np.arange(max_iters), valid_loss)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(["Training Loss", "Validation Loss"])
plt.title("Loss Curve")
plt.show()

# Q5.3.1
import matplotlib.pyplot as plt
# visualize some results
##########################
##### your code here #####
##########################
v_hidden_layer1 = forward(valid_x, params, 'layer1', relu)
v_hidden_layer2 = forward(v_hidden_layer1, params, 'layer2', relu)
v_hidden_layer3 = forward(v_hidden_layer2, params, 'layer3', relu)
v_probs = forward(v_hidden_layer3, params, 'output', sigmoid) # Probabilities

# Plot -> Select 5 classes from the 36 classes in the validation data and plot 2 validation images from each class
for i in range(5):
    plt.subplot(2, 1, 1)
    idx = int(3600/5*i+1)
    plt.imshow(valid_x[idx, :].reshape(32, 32).T)
    plt.subplot(2, 1, 2)
    plt.imshow(v_probs[idx, :].reshape(32, 32).T)
    plt.show()
    plt.subplot(2, 1, 1)
    idx = int(3600/5*i+2)
    plt.imshow(valid_x[idx, :].reshape(32, 32).T)
    plt.subplot(2, 1, 2)
    plt.imshow(v_probs[idx, :].reshape(32, 32).T)
    plt.show()

# Q5.3.2
from skimage.metrics import peak_signal_noise_ratio
# evaluate PSNR
##########################
##### your code here #####
##########################
noise_psnr = 0
for j in range(valid_x.shape[0]):
    noise_psnr += peak_signal_noise_ratio(valid_x[j, :], v_probs[j, :])
noise_psnr /= valid_x.shape[0]
print("PSNR: ", noise_psnr)
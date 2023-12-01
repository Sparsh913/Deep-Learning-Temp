import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None

    ##########################
    ##### your code here #####
    var = 2 / (in_size + out_size)
    bound = np.sqrt(3 * var)
    W = np.random.uniform(-bound, bound, (in_size, out_size))
    # print("W shape: ", W.shape)
    b = np.zeros(out_size)
    ##########################

    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = None

    ##########################
    ##### your code here #####
    # print("Shape x", x.shape)
    res = 1 / (1 + np.exp(-x))
    ##########################

    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]


    ##########################
    ##### your code here #####
    pre_act = X @ W + b
    post_act = activation(pre_act)
    ##########################


    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = None

    ##########################
    ##### your code here #####
    n, d = x.shape
    res = np.zeros((n, d))
    for i in range(n):
        temp = x[i, :]
        max_val = np.max(temp)
        temp = temp - max_val
        temp = np.exp(temp)
        sum = np.sum(temp)
        res[i, :] = temp / sum
    ##########################

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None

    ##########################
    ##### your code here #####
    loss = -np.sum(y * np.log(probs))
    # print("Shape Loss: ", loss.shape)
    n,d = y.shape
    acc = 0
    for i in range(n):
        idx = np.argmax(probs[i, :])
        if y[i,idx] == 1:
            acc += 1
    acc /= n
    ##########################

    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    # then compute the derivative W,b, and X
    ##########################
    ##### your code here #####
    # delt = delta * activation_deriv(post_act)
    n,d = delta.shape
    grad_A = activation_deriv(post_act)*delta
    grad_W = np.matmul(X.T,grad_A)
    grad_X = np.matmul(grad_A,W.T)
    grad_b = np.matmul(np.ones((1,n)),grad_A).flatten()
    ##########################

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    ##########################
    ##### your code here #####
    # n = x.shape[0]
    # indices = np.arange(n)
    # np.random.shuffle(indices)
    # x = x[indices]
    # y = y[indices]
    # for i in range(0, n, batch_size):
    #     batches.append((x[i:i+batch_size], y[i:i+batch_size]))

    random_batches = np.array_split(np.random.permutation(x.shape[0]), x.shape[0]//batch_size)
    
    for i in random_batches:
        #  batches.append((x[i,:],y[i,:]))
        batches.append((x[i, ...], y[i, ...]))

    ##########################
    return batches

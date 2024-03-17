"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys
import time

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    # read image file
    with gzip.open(image_filename, 'rb') as fi:
        image_content = fi.read()

    image_magic = struct.unpack('>I', image_content[:4])[0]
    if image_magic != 2051 :
        print("read file format error!")

    image_num = struct.unpack('>I', image_content[4:8])[0]
    image_row = struct.unpack('>I', image_content[8:12])[0]
    image_col = struct.unpack('>I', image_content[12:16])[0]

    ### vectorizaton form, very quick
    # Reshape image_content into a 3D array (image_num, image_row, image_col)
    image_content_reshaped = np.frombuffer(image_content, dtype=np.uint8, offset=16).reshape(image_num, image_row, image_col)
    # Convert uint8 values to float32 and normalize
    X = image_content_reshaped.astype(np.float32) / 255.0
    # Reshape X into a 2D array (image_num, image_row * image_col)
    X = X.reshape(image_num, -1)

    # --------------------------------------------------------------------------------------#

    # read label file
    with gzip.open(label_filename, 'rb') as fl:
        label_content = fl.read()

    label_magic = struct.unpack('>I', label_content[:4])[0]
    if label_magic != 2049 :
        print("read file format error!")

    label_num = struct.unpack('>I', label_content[4:8])[0]
    y = np.zeros(label_num, dtype=np.uint8)

    # get y from label_content
    for i in range (label_num):
        label = label_content[8 + i]
        y[i] = np.uint8(label)
    
    # --------------------------------------------------------------------------------------#

    return (X, y)
    ### END YOUR SOLUTION

# definde by mingxi
def softmax(tensor):
    tensor_exp = ndl.exp(tensor)
    tensor_exp_sum = ndl.summation(tensor_exp, (1,))
    tensor_exp_sum_expand = ndl.broadcast_to(ndl.reshape(tensor_exp_sum, (tensor.shape[0], 1)), tensor.shape)

    return  tensor_exp / tensor_exp_sum_expand 

# defined by mingxi
def relu_mask(tensor):
    condition = tensor.cached_data > 0
    return ndl.Tensor(condition.astype(int))

def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    Z_exp = ndl.exp(Z)
    Z_exp_sum_log_sum = ndl.summation(ndl.log(ndl.summation(Z_exp, (1,))))
    Z_y_sum = ndl.summation(Z * y_one_hot)

    return  (Z_exp_sum_log_sum - Z_y_sum) / Z.shape[0]
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION    
    m = X.shape[0]
    n = X.shape[1]
    k = W2.shape[1]

    iter_num = int(m / batch)

    I_y = np.zeros((m, k), dtype=np.uint8)
    I_y[np.arange(m), y] = 1

    for i in range(iter_num):
        X_batch = ndl.Tensor(X[i*batch:(i+1)*batch, :])
        Iy_batch = ndl.Tensor(I_y[i*batch:(i+1)*batch, :]) 

        Z = ndl.matmul(ndl.relu(ndl.matmul(X_batch, W1)), W2)
        loss_ce = softmax_loss(Z, Iy_batch)
        loss_ce.backward()
        
        
        W1 = ndl.Tensor(W1.numpy() - W1.grad.numpy() * lr) 
        W2 = ndl.Tensor(W2.numpy() - W2.grad.numpy() * lr) 
        #W1.data = W1.data + W1.grad * (-lr) 
        #W2.data = W2.data + W2.grad * (-lr) 

    return W1, W2
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)

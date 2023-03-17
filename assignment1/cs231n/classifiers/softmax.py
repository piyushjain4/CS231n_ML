from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_class = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = np.dot(X,W)
    scores -=np.reshape(np.max(scores,axis =1),(num_train,1))
    f = np.sum(np.exp(scores),axis=1)
    for i in range (num_train):
      sum=0
      for j in range (num_class):
        dW[:,j:j+1] += (((X.T)[:,i:i+1])*(np.exp(scores[i][j])))/f[i]
      dW[:,y[i]:y[i]+1] -= ((X.T)[:,i:i+1])
      loss += np.log(f[i]) - scores[i][y[i]]


    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += (2*reg)*(np.sum(W,axis=0))


    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_class = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = np.dot(X,W)
    scores -=np.reshape(np.max(scores,axis =1),(num_train,1))
    f = (np.exp(scores))/(np.reshape((np.sum(np.exp(scores),axis=1)),(num_train,1)))
    a = f[[np.arange(num_train)],y]
    loss= (-np.log(a))
    loss = (np.sum(loss))/num_train
    f[[np.arange(num_train)],y] -=1
    dW = ((X.T).dot(f))
    dW /= num_train
    loss += reg * np.sum(W * W)
    dW += (2*reg)*(np.sum(W,axis=0))


    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

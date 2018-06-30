#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(X, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    the backward propagation for the gradients for all parameters.

    Notice the gradients computed here are different from the gradients in
    the assignment sheet: they are w.r.t. weights, not inputs.

    Arguments:
    X -- M x Dx matrix, where each row is a training example x.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Note: compute cost based on `sum` not `mean`.
    ### YOUR CODE HERE: forward propagation
    h = sigmoid(np.dot(X, W1) + b1)
    yhat = softmax(np.dot(h, W2) + b2)
    cross_entropy = -np.log(yhat)[labels == 1]
    cost = np.sum(cross_entropy)  # / len(labels)
    ### END YOUR CODE   

    # Things look too good to be true... Tried twice and the gradient check passed.
    ### YOUR CODE HERE: backward propagation
    dl_dyhat = (-1/yhat)[labels == 1]   # m x 1, m the number of points;
    dyhat_dsoftmax = yhat * (1 - yhat)  # m x n, n the number of classes;
    dl_dsoftmax = dyhat_dsoftmax * np.reshape(dl_dyhat, (-1, 1))
    gradW2 = np.reshape(np.sum(h,0),[-1,1])*np.reshape(np.sum(dl_dsoftmax,0),[1,-1]) #np.dot(h.T, dl_dsoftmax)  # n x h, transpose shape of W2;
    gradb2 = np.reshape(np.sum(dl_dsoftmax, axis=0), (1,-1))  # n x 1   
    dl_dh = np.dot(np.sum(dl_dsoftmax, axis=0), W2.T)  # m x h, sum up all m
    dh_dsigmoid = np.sum(sigmoid_grad(h),0)   # m x h, sumup all m
    dl_dsigmoid = dl_dh * dh_dsigmoid
    gradW1 = np.reshape(np.sum(X,0),[-1,1])*np.reshape(dl_dsigmoid,[1,-1])
    gradb1 = np.reshape(dl_dsigmoid, (1,-1))
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print ("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), data)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print ("Running your sanity checks...")
    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()

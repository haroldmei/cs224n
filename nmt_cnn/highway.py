#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h

import torch
import torch.nn as nn

class Highway(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, dropout_rate=0.2):
        """
        """
        ### the final word embedding size
        self.embed_size = embed_size
        self.dropout_rate = dropout_rate

        ### input output size are both embed_size
        self.W_proj = nn.Linear(self.embed_size, self.embed_size, bias=True)
        self.W_gate = nn.Linear(self.embed_size, self.embed_size, bias=True)
        self.Dropout = nn.Dropout(self.dropout_rate)

    def forward(self, input):
        """
        params:
        input: a batch of words with shape (batch, embed_size)
        """
        xproj = nn.ReLU(self.W_proj(input))
        xgate = nn.Sigmoid(self.W_proj(input))
        x_highway = xproj * xgate +(1 - xgate) * input
        
        return x_highway


### END YOUR CODE 


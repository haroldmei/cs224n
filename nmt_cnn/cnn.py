#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn

class CNN(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, emb_char = 50, kernel=5, emb_word=100):
        """
        """
        ### the final word embedding size
        self.kernel = kernel
        self.emb_word = emb_word
        self.emb_char = emb_char
        self.M_WORD = 21

        ### input output size are both embed_size
        self.Conv1d = nn.Conv1d(self.emb_char * self.M_WORD, self.emb_word, self.emb_char * self.kernel, bias=True)

    def forward(self, input):
        """
        params:
        input: a batch of words with shape (batch, embed_size)
        """
        word_emb = self.Conv1d(input)
        return word_emb


### END YOUR CODE


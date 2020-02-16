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
        super(CNN, self).__init__()
        self.kernel = kernel
        self.emb_word = emb_word
        self.emb_char = emb_char
        self.max_words = 21
        ### input output size are both embed_size
        self.Conv1d = \
            nn.Sequential(
                # input: batch * emb_char * max_words
                # output: batch * emb_word * (max_words - kernel + 1)
                nn.Conv1d(self.emb_char, self.emb_word, self.kernel, bias=True),
                nn.ReLU(),
                nn.MaxPool1d(self.max_words - self.kernel + 1)
            )

    def forward(self, input):
        """
        params:
        input: a batch of words with shape (batch, emb_char, max_words)
        output: (batch, emb_word)
        """
        word_emb = self.Conv1d(input).squeeze(2)
        return word_emb


### END YOUR CODE


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j

        self.dropout_rate = 0.3
        self.emb_char = 50
        self.embed_size = embed_size
        pad_token_idx = vocab.char2id['<pad>']
        self.embeddings = nn.Embedding(len(vocab.char2id), self.emb_char, padding_idx=pad_token_idx)
        # Conv1d + Pooling1d
        cnn = CNN(emb_char=self.emb_char, emb_word = self.embed_size) 
        # with dropout rate? 
        highway = Highway(embed_size = self.embed_size, dropout_rate=self.dropout_rate)
        # with a dropout layer?
        dropout = nn.Dropout(self.dropout_rate)
        # chain all toghether
        self.construct_emb = nn.Sequential(cnn, highway, dropout)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j

        print(input.shape)
        char_embeddings = self.embeddings(input) # sentence_length, batch_size, max_word_length,
        sent_len, batch_size, max_word, _ = char_embeddings.shape
        view_shape = (sent_len * batch_size, max_word, self.emb_char)
        char_embeddings = char_embeddings.view(view_shape).transpose(1, 2)
        emb_output = self.construct_emb(char_embeddings)
        emb_output = emb_output.view(sent_len, batch_size, self.embed_size)
        #print(view_shape)
        return emb_output


        ### END YOUR CODE


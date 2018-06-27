#! /usr/bin/env python

import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
from datetime import datetime
from utils import *
import tensorflow as tf

nltk.download("book")

_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '8000'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '80'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
_NEPOCH = int(os.environ.get('NEPOCH', '100'))
_MODEL_FILE = os.environ.get('MODEL_FILE')
SKIP_STEP = 10

#
def vanillaRNN(model, X_train, y_train, learning_rate=0.005, nepoch=1):
    loss = model.calculate_total_loss(X_train, y_train)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    with tf.Session() as sess:
        # Step 6: initialize iterator and variables
        sess.run(tf.global_variables_initializer())
        total_loss = 0.0 # we use this to calculate late average loss in the last SKIP_STEP steps
        writer = tf.summary.FileWriter('graphs/word2vec_simple', sess.graph)
        for epoch in range(nepoch):
            loss_batch, _ = sess.run([loss, optimizer])
            total_loss += loss_batch
            #if (epoch + 1) % SKIP_STEP == 0:
            print('Average loss at step {}: {:5.1f}'.format(epoch, total_loss / SKIP_STEP))
            #total_loss = 0.0

vocabulary_size = _VOCABULARY_SIZE
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# Read the data and append SENTENCE_START and SENTENCE_END tokens
print ("Reading CSV file...")
with open('data/reddit-comments-2015-08.csv', 'rt', encoding="utf8") as f:
    reader = csv.reader(f, skipinitialspace=True)
    next(reader)
    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]

print ("Parsed %d sentences." % (len(sentences)))
    
# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print ("Found %d unique words tokens." % len(word_freq.items()))

# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

print ("Using vocabulary size %d." % vocabulary_size)
print ("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
MAX_LEN=np.max([len(X_train[i]) for i in np.arange(len(y_train))])


word_dim = vocabulary_size
hidden_dim = _HIDDEN_DIM

#model = RNNNumpy(vocabulary_size, hidden_dim=_HIDDEN_DIM)
#train_with_sgd(model, X_train, y_train)

V = tf.get_variable('RNNTF_V', shape=[word_dim, hidden_dim],   dtype=tf.float64, initializer=tf.random_uniform_initializer(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim)))

UI = tf.get_variable('RNNTF_UI', shape=[hidden_dim, word_dim],   dtype=tf.float64, initializer=tf.random_uniform_initializer(-np.sqrt(1./word_dim),   np.sqrt(1./word_dim)))
UF = tf.get_variable('RNNTF_UF', shape=[hidden_dim, word_dim],   dtype=tf.float64, initializer=tf.random_uniform_initializer(-np.sqrt(1./word_dim),   np.sqrt(1./word_dim)))
UG = tf.get_variable('RNNTF_UG', shape=[hidden_dim, word_dim],   dtype=tf.float64, initializer=tf.random_uniform_initializer(-np.sqrt(1./word_dim),   np.sqrt(1./word_dim)))
UO = tf.get_variable('RNNTF_UO', shape=[hidden_dim, word_dim],   dtype=tf.float64, initializer=tf.random_uniform_initializer(-np.sqrt(1./word_dim),   np.sqrt(1./word_dim)))
WI = tf.get_variable('RNNTF_WI', shape=[hidden_dim, hidden_dim], dtype=tf.float64, initializer=tf.random_uniform_initializer(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim)))
WF = tf.get_variable('RNNTF_WF', shape=[hidden_dim, hidden_dim], dtype=tf.float64, initializer=tf.random_uniform_initializer(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim)))
WG = tf.get_variable('RNNTF_WG', shape=[hidden_dim, hidden_dim], dtype=tf.float64, initializer=tf.random_uniform_initializer(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim)))
WO = tf.get_variable('RNNTF_WO', shape=[hidden_dim, hidden_dim], dtype=tf.float64, initializer=tf.random_uniform_initializer(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim)))

S0 = tf.constant(0.0, tf.float64, shape=[hidden_dim, 1])
learning_rate = tf.placeholder(tf.float32, shape=[])
NumWords = np.sum((len(y_i) for y_i in y_train))

datasetX = tf.data.Dataset.from_generator(lambda: X_train, tf.int32, output_shapes=[None]).repeat()
iteratorX = datasetX.make_one_shot_iterator()
next_element_X = iteratorX.get_next()
datasetY = tf.data.Dataset.from_generator(lambda: y_train, tf.int32, output_shapes=[None]).repeat()
iteratorY = datasetY.make_one_shot_iterator()
next_element_Y = iteratorY.get_next()

condVanilla = lambda pred, index, s, : tf.less(index, tf.size(next_element_Y))
def bodyVanilla(pred, index, s):
  s = tf.nn.tanh(tf.reshape(UI[:,next_element_X[i]], [-1,1]) + tf.matmul(WI, s))
  o = tf.nn.softmax(tf.reshape(tf.matmul(V, s),([-1])))
  pred += tf.log(o[next_element_Y[i]])
  return pred, tf.add(i, 1), s

condLSTM = lambda pred, index, s, c: tf.less(index, tf.size(next_element_Y))
def bodyLSTM(pred, index, s, c):
   i = tf.nn.sigmoid(tf.reshape(UI[:,next_element_X[index]], [-1,1]) + tf.matmul(WI, s))
   f = tf.nn.sigmoid(tf.reshape(UF[:,next_element_X[index]], [-1,1]) + tf.matmul(WF, s))
   o = tf.nn.sigmoid(tf.reshape(UO[:,next_element_X[index]], [-1,1]) + tf.matmul(WO, s))
   g = tf.nn.tanh(tf.reshape(UG[:,next_element_X[index]], [-1,1]) + tf.matmul(WG, s))
   c = tf.multiply(c, f) + tf.multiply(g, i)
   s = tf.multiply(tf.nn.tanh(c), o)
   oo = tf.nn.softmax(tf.reshape(tf.matmul(V, s),([-1])))
   pred += tf.log(oo[next_element_Y[index]])
   return pred, tf.add(index, 1), s, c

def rnnLoss(c,b):
  pred = np.float64(0.0)
  r = tf.while_loop(c, b, [pred, 0, S0, S0])
  L = -r[0]
  return L#pred, o, s

loss = rnnLoss(condLSTM, bodyLSTM)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter('graphs/rnn_lstm', sess.graph)
learningRate = _LEARNING_RATE
totalLoss = 0.0

for idx in range(_NEPOCH):
  index = 0
  total_loss = 0
  while index < (len(y_train)):
      loss_batch, _ = sess.run([loss, optimizer], feed_dict={learning_rate: learningRate})
      total_loss += loss_batch
      index += 1
      #if index % 5000 == 0:
      #    time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
      #    print ("%s: idx =%d epoch=%d: %.10f" % (time, index, idx, loss_batch))
  total_loss /= NumWords
  if totalLoss < total_loss:
      print ("Learning rate set to %.10f, loss=%.10f, new loss=%.10f" % (learningRate, totalLoss, total_loss))
      learningRate /= 2
  totalLoss = total_loss
  ui,uf,ug,uo,wi,wf,wg,wo,v = sess.run([UI,UF,UG,UO,WI,WF,WG,WO,V])
  time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
  save_parameters_tf("./data/rnn-lstm-%d-%d-%s-%f.npz" % (hidden_dim, word_dim, time, total_loss), ui,uf,ug,uo,wi,wf,wg,wo,v)
  print ("%s: epoch=%d: %f" % (time, idx, total_loss))


#for idx in range(_NEPOCH):
#    total_loss, _ = sess.run([loss, optimizer], feed_dict={learning_rate: learningRate})
#    total_loss /= NumWords
#    if totalLoss < total_loss:
#        print ("Learning rate set to %.10f, loss=%.10f, new loss=%.10f" % (learningRate, totalLoss, total_loss))
#        learningRate /= 2
#    totalLoss = total_loss
#    time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#    print ("%s: epoch=%d: %.10f" % (time, idx, total_loss))

#UU,VV,WW=load_parameters_tf("./data/rnn-tf-80-8000-2018-06-17-17-20-09-4.985381.npz")
UUI,UUF,UUG,UUO,WWI,WWF,WWG,WWO,VV=load_parameters_tf("./data/rnn-lstm-80-8000-2018-06-22-08-42-07-4.242547.npz")
#UUI,UUF,UUO,UUG,WWI,WWF,WWO,WWG,VV = sess.run([UI,UF,UO,UG,WI,WF,WO,WG,V])

def forward_propagation(x):
  # The total number of time steps
  T = len(x)
  # During forward propagation we save all hidden states in s because need them later.
  # We add one additional element for the initial hidden, which we set to 0
  s = np.zeros((T + 1, hidden_dim))
  s[-1] = np.zeros(hidden_dim)
  c = np.zeros((T + 1, hidden_dim))
  c[-1] = np.zeros(hidden_dim)
  # The outputs at each time step. Again, we save them for later.
  oo = np.zeros((T, word_dim))
  # For each time step...
  for t in np.arange(T):
      # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
       i = sigmoid(UUI[:,x[t]] + WWI.dot(s[t-1]))
       f = sigmoid(UUF[:,x[t]] + WWF.dot(s[t-1]))
       o = sigmoid(UUO[:,x[t]] + WWO.dot(s[t-1]))
       g = np.tanh(UUG[:,x[t]] + WWG.dot(s[t-1]))
       c[t] = c[t - 1] * f + g * i
       s[t] = np.tanh(c[t]) * o
       #s[t] = np.tanh(UU[:,x[t]] + WW.dot(s[t-1]))
       oo[t] = softmax(VV.dot(s[t]))
  return [oo, s]

def generate_sentence():
  # We start the sentence with the start token
  new_sentence = [word_to_index[sentence_start_token]]
  # Repeat until we get an end token
  len = 20
  while not new_sentence[-1] == word_to_index[sentence_end_token]:
      if len < 0:
          break
      len = len - 1
      next_word_probs,_ = forward_propagation(new_sentence)
      sampled_word = word_to_index[unknown_token]
      # We don't want to sample unknown words
      while sampled_word == word_to_index[unknown_token]:
          samples = np.random.multinomial(1, next_word_probs[-1])
          sampled_word = np.argmax(samples)
      new_sentence.append(sampled_word)
  sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
  return sentence_str

num_sentences = 50
senten_min_length = 7
for i in range(num_sentences):
  sent = []
  # We want long sentences, not sentences with one or two words
  while len(sent) < senten_min_length:
      sent = generate_sentence()
  print (" ".join(sent))




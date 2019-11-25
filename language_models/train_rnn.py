import numpy as np
from utils import *
from load_data import *

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

# use tf.while_loop to iterate through all sentences.
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
totalLoss = 0.0

for idx in range(epoch):
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
  save_parameters_lstm("./data/rnn-lstm-%d-%d-%s-%f.npz" % (hidden_dim, word_dim, time, total_loss), ui,uf,ug,uo,wi,wf,wg,wo,v)
  print ("%s: epoch=%d: %f" % (time, idx, total_loss))


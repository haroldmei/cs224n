import numpy as np
from utils import *
from load_data import *

# load pre-trained models
UUI, UUF, UUG, UUO, WWI, WWF, WWG, WWO, VV = load_parameters_lstm("./data/rnn-lstm-80-8000-2018-06-22-08-42-07-4.242547.npz")
U,V,W = load_parameters_vanilla("./data/rnn-tf-80-8000-2018-06-17-19-48-53-4.929756.npz")

def forward_propagation_lstm(x):
    T = len(x)
    s = np.zeros((T + 1, hidden_dim))
    s[-1] = np.zeros(hidden_dim)
    c = np.zeros((T + 1, hidden_dim))
    c[-1] = np.zeros(hidden_dim)
    oo = np.zeros((T, word_dim))
    # For each time step...
    for t in np.arange(T):
        # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
        i = sigmoid(UUI[:, x[t]] + WWI.dot(s[t-1]))
        f = sigmoid(UUF[:, x[t]] + WWF.dot(s[t-1]))
        o = sigmoid(UUO[:, x[t]] + WWO.dot(s[t-1]))
        g = np.tanh(UUG[:, x[t]] + WWG.dot(s[t-1]))
        c[t] = c[t - 1] * f + g * i
        s[t] = np.tanh(c[t]) * o
        oo[t] = softmax(VV.dot(s[t]))
    return [oo, s]


def forward_propagation_vanilla(x):
    T = len(x)
    s = np.zeros((T + 1, hidden_dim))
    s[-1] = np.zeros(hidden_dim)
    oo = np.zeros((T, word_dim))
    # For each time step...
    for t in np.arange(T):
        # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
        s[t] = np.tanh(U[:,x[t]] + W.dot(s[t-1]))
        oo[t] = softmax(V.dot(s[t]))
    return [oo, s]


# this can be done by reusing the tf session as well
def generate_sentence(prop):
    # We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    # Repeat until we get an end token
    len = 20
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        if len < 0:
            break
        len = len - 1
        next_word_probs, _ = prop(new_sentence)
        sampled_word = word_to_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str


num_sentences = 20
senten_min_length = 7
print ("\n################################ GENERATE WITH VANILLA RNN ################################\n")
# generate sentences with vanilla rnn model
for i in range(num_sentences):
    sent = []
    # We want long sentences, not sentences with one or two words
    while len(sent) < senten_min_length:
        sent = generate_sentence(forward_propagation_vanilla)
    print(" ".join(sent))


print ("\n################################ GENERATE WITH LSTM RNN ################################\n")
# generate sentences with lstm rnn model
for i in range(num_sentences):
    sent = []
    # We want long sentences, not sentences with one or two words
    while len(sent) < senten_min_length:
        sent = generate_sentence(forward_propagation_lstm)
    print(" ".join(sent))

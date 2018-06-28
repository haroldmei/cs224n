import numpy as np

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

def sigmoid(x):
   return 1 / (1 + np.exp(-x))

def save_parameters_vanilla(outfile, U,V,W):
    np.savez(outfile, U=U,V=V,W=W)
    print ("Saved tf parameters to %s." % outfile)

def load_parameters_vanilla(path):
    npzfile = np.load(path)
    U,V,W = npzfile["U"], npzfile["V"], npzfile["W"]
    print ("Loaded tf parameters from %s. hidden_dim=%d word_dim=%d" % (path, U.shape[0], U.shape[1]))
    return U,V,W

def save_parameters_lstm(outfile, UI,UF,UG,UO,WI,WF,WG,WO,V):
    np.savez(outfile, UI=UI,UF=UF,UG=UG,UO=UO,WI=WI,WF=WF,WG=WG,WO=WO,V=V)
    print ("Saved tf parameters to %s." % outfile)

def load_parameters_lstm(path):
    npzfile = np.load(path)
    UI,UF,UO,UG,WI,WF,WO,WG,V = npzfile["UI"], npzfile["UF"], npzfile["UG"], npzfile["UO"], npzfile["WI"], npzfile["WF"], npzfile["WG"], npzfile["WO"], npzfile["V"]
    print ("Loaded tf parameters from %s. hidden_dim=%d word_dim=%d" % (path, UI.shape[0], UI.shape[1]))
    return UI,UF,UO,UG,WI,WF,WO,WG,V
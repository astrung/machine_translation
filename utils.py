import numpy as np

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

def save_model_parameters_theano(outfile, model):
    U, V, W = model.U.get_value(), model.V.get_value(), model.W.get_value()
    np.savez(outfile, U=U, V=V, W=W)
    print("Saved model parameters to %s." % outfile)
   
def load_model_parameters_theano(path, model):
    npzfile = np.load(path)
    U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
    model.hidden_dim = U.shape[0]
    model.word_dim = U.shape[1]
    model.U.set_value(U)
    model.V.set_value(V)
    model.W.set_value(W)
    print("Loaded model parameters from %s. hidden_dim=%d word_dim=%d" % (path, U.shape[0], U.shape[1]))

def euclidean_error(input, label):
    return label - input


def crossentropy_error(input, label):
    return label - input


def sigm(x):
    return 1 / (1 + np.exp(-x))


def dsigm(x):
    return x * (1 - x)


def tanh(x):
    return np.tanh(x)


def dtanh(x):
    return (1 - np.square(x))


def relu(x):
    return np.maximum(0, x)


def drelu(x):
    return np.array(x > 0, dtype=int)
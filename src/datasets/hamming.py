import hamming_codec
import numpy as np
import pickle
import os

def get_hamming(name, path='data/hamming_codec.pkl'):
    if os.path.exists(path):
        data = pickle.load(open(path, 'rb'))
        return data
    # generate data
    xs = np.arange(0, 2**10)
    # get ys
    ys
    data = (xs, ys)
    with open(path, 'wb') as f:
        pickle.dump(data, f)

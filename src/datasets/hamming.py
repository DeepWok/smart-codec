import hamming_codec
import pickle
import os

def get_hamming(name, path='data/hamming_codec.pkl'):
    if os.path.exists(path):
        data = pickle.load(open(path, 'rb'))
        return data
    with open(path, 'wb') as f:
        data = hamming_codec.get_hamming(name)
        pickle.dump(data, f)

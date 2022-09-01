import torch
import tqdm
import hamming_codec
import numpy as np
import pickle
import os


def bin2vec(x):
    return np.array([int(i) for i in x])


def get_hamming(path='data/hamming_codec', bits=16, size=6000000, split=(0.8, 0.2)):
    path = path + f'/processed_bits{bits}/'
    if os.path.exists(path):
        train_data = pickle.load(open(path + 'train.pkl', 'rb'))
        val_data = pickle.load(open(path + 'val.pkl', 'rb'))
        test_data = pickle.load(open(path + 'test.pkl', 'rb'))
        return (train_data, val_data, test_data)

    # generate data
    xs = np.random.randint(0, 2**bits-1, size=size)
    # get ys
    ys = []
    print('Start input generation...')
    for x in tqdm.tqdm(xs):
        y = hamming_codec.encode(x, bits)
        ys.append(y)
    print('Input generation finished.')
    
    train_length = int(size * split[0])
    val_length = int(size * split[1])

    train_xs, train_ys = xs[:train_length], ys[:train_length]
    val_xs, val_ys = xs[train_length:train_length+val_length], ys[train_length:train_length+val_length]
    test_xs, test_ys = xs[train_length+val_length:], ys[train_length+val_length:]

    os.makedirs(path)
    pickle.dump((train_xs, train_ys), open(path + 'train.pkl', 'wb'))
    pickle.dump((val_xs, val_ys), open(path + 'val.pkl', 'wb'))
    pickle.dump((test_xs, test_ys), open(path + 'test.pkl', 'wb'))
    return ((train_xs, train_ys), (val_xs, val_ys), (test_xs, test_ys))


class HammingDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, mode='train', path='data/hamming_codec', bits=16, size=6000000, split=(0.8, 0.2)):
        train_data, val_data, test_data = get_hamming(path=path, bits=bits, size=size, split=split)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.mode = mode
        self.transform = transform
        self.bits = bits
        if self.mode == 'train':
            self.xs, self.ys = self.train_data
        elif self.mode == 'val':
            self.xs, self.ys = self.val_data
        elif self.mode == 'test':
            self.xs, self.ys = self.test_data
        else:
            raise ValueError(f'{self.mode} is not found')

    def __len__(self):
        return len(self.xs)
    
    def _transform_x(self, x):
        binaries = "{0:b}".format(x)
        binaries = bin2vec(binaries)
        if len(binaries) < self.bits:
            zeros = np.zeros(self.bits - len(binaries)).astype(int)
            binaries = np.concatenate((zeros, binaries))
        return binaries

    def __getitem__(self, idx):
        x = self.xs[idx]
        bin_x = self._transform_x(x)
        y = self.ys[idx]
        bin_y = bin2vec(y)
        bin_x = torch.from_numpy(bin_x).type(torch.FloatTensor)
        bin_y = torch.from_numpy(bin_y).type(torch.FloatTensor)
        return bin_x, bin_y
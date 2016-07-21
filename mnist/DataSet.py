from urllib.request import urlretrieve
import numpy as np
import gzip
from collections import namedtuple
import os

DataSet = namedtuple('DataSet', 'images labels')

WORK_DIR = 'data'
SOURCE = 'http://yann.lecun.com/exdb/mnist/'
NUM_CHANNELS = 1
WIDTH = 28
HEIGHT = 28

def download(filename):
    if not os.path.exists(WORK_DIR):
        os.mkdir(WORK_DIR)

    if not os.path.exists(os.path.join(WORK_DIR, filename)):
        print("Downloading " + filename)
        urlretrieve(SOURCE + filename, os.path.join(WORK_DIR, filename))

def read_images(filename):
    download(filename)

    with gzip.open(os.path.join(WORK_DIR, filename), 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)

    data = data.reshape(-1, WIDTH, HEIGHT, NUM_CHANNELS)

    return data / np.float32(256)

def read_labels(filename):
    download(filename)

    with gzip.open(os.path.join(WORK_DIR, filename), 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)

    onehot = np.zeros((len(data), 10))

    for idx, cls in enumerate(data):
        onehot[idx][cls] = 1

    return onehot

TRAIN_DATASET = DataSet(
        images=read_images('train-images-idx3-ubyte.gz'),
        labels=read_labels('train-labels-idx1-ubyte.gz'),
    )

TEST_DATASET = DataSet(
        images=read_images('t10k-images-idx3-ubyte.gz'),
        labels=read_labels('t10k-labels-idx1-ubyte.gz'),
    )

def iter_batches(batch_size, dataset=TRAIN_DATASET):
    indices = np.arange(len(dataset.images))
    np.random.shuffle(indices)

    for start_idx in range(0, len(dataset.images) - batch_size + 1, batch_size):
        idxs = indices[start_idx:start_idx + batch_size]
        yield dataset.images[idxs], dataset.labels[idxs]

import pandas as pd
from data_loader import load_data
import numpy as np
from tqdm import tqdm
import time

read_time = 1
process_time = 2
buffer_pool_length = 1000

# data, vocab = load_data("data/trace.txt")
X = np.loadtxt("data/index_trace.csv").astype(int)
vocab = sorted(list(set(X)))

def LRU(X_prefetch=None):
    buffer_pool = []
    buffer_pool_set = set()
    hit_count = 0
    tic = time.time()

    def insert(x):
        if x in buffer_pool_set:
            buffer_pool_set.remove(x)
            buffer_pool.remove(x)
            buffer_pool.append(x)
            buffer_pool_set.add(x)
        else:
            buffer_pool.append(x)
            buffer_pool_set.add(x)

            if len(buffer_pool) > buffer_pool_length:
                buffer_pool_set.remove(buffer_pool[0])
                buffer_pool.pop(0)

    num_iter = 0

    for x in tqdm(X):
        if x in buffer_pool_set:
            hit_count += 1

        insert(x)

        if X_prefetch is not None:
            for xr in X_prefetch[num_iter]:
                insert(xr)

        num_iter += 1

    hit_rate = hit_count / len(X)
    return  hit_rate

def simulate_random():
    result = []
    column = []

    for k in [0, 10, 20, 50]:
        if k == 0:
            lru_hit_rate = LRU()
        else:
            np.random.seed(1)
            X_prefetch = np.random.choice(vocab, size=(len(X), k))
            lru_hit_rate = LRU(X_prefetch)

        column.append(str(k))
        result.append(lru_hit_rate)

    result = pd.DataFrame([result], columns=column)
    result.to_csv("random_hit_rate.csv", index=False)


def simulate_RNN():
    result = []
    column = []

    for k in [10, 20, 50]:
        rnn_result = np.loadtxt("all_preds/top-{}.csv".format(k), delimiter=",").astype(int)
        X_prefetch = rnn_result[:, :-1]
        print(k, X_prefetch.shape)
        hit_rate = RNN(X_prefetch)
        column.append(str(k))
        result.append(hit_rate)

    result = pd.DataFrame([result], columns=column)
    result.to_csv("RNN_hit_rate.csv", index=False)


simulate_random()
# simulate_RNN()


# def RNN(k):





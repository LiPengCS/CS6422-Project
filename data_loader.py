import torch
import numpy as np
import pandas as pd
from torchtext.legacy.data import Field, Dataset, BucketIterator, Example

class DataFrameDataset(Dataset):
    def __init__(self, df, fields):
        examples = []
        for i, row in df.iterrows():
            examples.append(Example.fromlist([row.X, row.Y, row.is_read, row.is_miss], fields))
        super().__init__(examples, fields)

def load_data(file_path):
    vocab = set()
    data = {
        "X": [],
        "Y": [],
        "is_read": [],
        "is_miss": []
    }

    pre_page_id = None
    pre_read_write = None
    with open(file_path, 'r') as f:
        for line in f.readlines():
            cur_page_id, _, cur_read_write, cur_miss_hit, _ = line.split(",")

            if pre_page_id is not None:
                #X, Y, is_read, is_miss
                data["X"].append(pre_page_id)
                data["Y"].append(cur_page_id)
                data["is_read"].append(int(pre_read_write == "read"))
                data["is_miss"].append(int(cur_miss_hit == "miss"))

            pre_page_id = cur_page_id
            pre_read_write = cur_read_write
            vocab.add(cur_page_id)

    vocab = list(sorted(vocab))
    return data, vocab

def split_windows(data, window_size=100):
    windows = {}
    for key, value in data.items():
        n_windows = len(value) // window_size
        new_value = []
        for i in range(n_windows):
            slice = value[i*window_size:(i+1)*window_size]
            new_value.append(slice)
        windows[key] = new_value
    return windows

def train_val_test_split(data, ratio=0.2, random_state=1, shuffle=True):
    N = len(data)
    if shuffle:
        np.random.seed(random_state)
        indices = np.random.permutation(N)
    else:
        indices = np.arange(N)

    train_val_size = int(N * (1-ratio))
    train_size = int(train_val_size * (1-ratio))

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_val_size]
    test_indices = indices[train_val_size:]

    train_data = data.iloc[train_indices].reset_index(drop=True)
    val_data = data.iloc[val_indices].reset_index(drop=True)
    test_data = data.iloc[test_indices].reset_index(drop=True)
    return train_data, val_data, test_data

def indexing(data, vocab, pretrained_embedding=None):
    if pretrained_embedding is None:
        pid_to_idx = {}
        for pid in vocab:
            pid_to_idx[pid] = len(pid_to_idx)
    else:
        pid_to_idx = pretrained_embedding.key_to_index

    X = [pid_to_idx[pid] for pid in data["X"]]
    Y = [pid_to_idx[pid] for pid in data["Y"]]
    data["X"] = X
    data["Y"] = Y
    return data, pid_to_idx

def get_data_loaders(file_path, params, pretrained_embedding=None, shuffle=True):
    X = Field(sequential=True, use_vocab=False, batch_first=True)
    Y = Field(sequential=True, use_vocab=False, batch_first=True)
    is_read = Field(sequential=True, use_vocab=False, batch_first=True)
    is_miss = Field(sequential=True, use_vocab=False, batch_first=True)
    fields = [('X', X), ('Y', Y), ('is_read', is_read), ('is_miss', is_miss)]

    data, vocab = load_data(file_path)
    data, pid_to_idx = indexing(data, vocab, pretrained_embedding)

    windows = split_windows(data, window_size=params["window_size"])
    windows = pd.DataFrame(windows, columns=["X", "Y", "is_read", "is_miss"])
    train_df, val_df, test_df = train_val_test_split(windows, random_state=params["split_seed"], shuffle=shuffle)

    train_ds = DataFrameDataset(train_df, fields)
    val_ds = DataFrameDataset(val_df, fields)
    test_ds = DataFrameDataset(test_df, fields)

    if shuffle:
        bucket_shuffle = None
    else:
        bucket_shuffle = False

    train_iter, val_iter, test_iter = BucketIterator.splits(
                                        (train_ds, val_ds, test_ds),
                                        batch_size=params["batch_size"],
                                        device=params["device"],
                                        shuffle=bucket_shuffle,
                                        sort=False)
    return  train_iter, val_iter, test_iter, pid_to_idx


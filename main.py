from model import RNN
from train import train_evaluate
import torch
import argparse
from gensim.models import Word2Vec
from data_loader import get_data_loaders
import os
import numpy as np
import json

parser = argparse.ArgumentParser()
parser.add_argument('--use_pre', action="store_true", default=False)
parser.add_argument('--path', default="data/trace.txt")
parser.add_argument('--result_dir', default="result")
parser.add_argument('--embed_path', default="embeddings/word2vec.embed")
args = parser.parse_args()

params = {
    "num_epochs": 500,
    # "num_epochs": 2,
    "lr": 0.001,
    "dropout": 0,
    "device": "cpu",
    "split_seed": 1,
    "seed": 1,
    "batch_size": 128,
    "window_size":100,
    "top_k": 50,
    "num_layers": 2,
    "embedding_dim": 50,
    "hidden_dim": 20,
    "use_pre": args.use_pre,
    "output_pred": False
}

if torch.cuda.is_available():
    params["device"] = "cuda"

for dropout in [0]:
    print("Dropout", dropout)
    params["dropout"] = dropout

    if args.use_pre:
        pretrained_embedding = Word2Vec.load(args.embed_path).wv
        result_dir = os.path.join(args.result_dir, "pretrained", "dropout_{}".format(dropout))
    else:
        pretrained_embedding = None
        result_dir = os.path.join(args.result_dir, "scratch", "dropout_{}".format(dropout))

    train_iter, val_iter, test_iter, pid_to_idx = get_data_loaders(args.path, params, pretrained_embedding=pretrained_embedding)
    params["vocab_size"] = len(pid_to_idx)

    # set random seed
    torch.manual_seed(params["seed"])
    np.random.seed(params["seed"])
    if "cuda" in params["device"]:
        torch.cuda.manual_seed(params["seed"])

    model = RNN(params, pretrained_embedding=pretrained_embedding).to(params["device"])

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    train_evaluate(model, train_iter, val_iter, test_iter, optimizer, params, log_dir=result_dir)

    with open(os.path.join(result_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=4)
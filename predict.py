from model import RNN
import torch
import time
import json
import os
from data_loader import get_data_loaders
import pandas as pd
from utils import makedir
import numpy as np
from data_loader import load_data
import sys
from tqdm import tqdm

topk = int(sys.argv[1])
result_dir = "result/scratch/dropout_0"
data_path = "data/trace.txt"

with open(os.path.join(result_dir, "params.json"), "r") as f:
    params = json.load(f)

if torch.cuda.is_available():
    params["device"] = "cuda"
else:
    params["device"] = "cpu"

# load model
model_path = os.path.join(result_dir, "best_model", "acc-top{}.model".format(topk))
model = RNN(params).to(params["device"])

if params["device"] == "cuda":
    model.load_state_dict(torch.load(model_path))
else:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

model.index_embedding(use_gpu= (params["device"]!="cpu"))
# load data

train_iter, val_iter, test_iter, pid_to_idx = get_data_loaders(data_path, params, shuffle=False)
# data, vocab = load_data(data_path)
# data_idx = []
# for x in data["X"]:
#     data_idx.append(pid_to_idx[x])
# np.savetxt("data/index_trace.csv", data_idx, delimiter=",")

def predict(data_iter):
    output_pred = []

    for batch in tqdm(data_iter):
        X = batch.X.to(params["device"])
        Y = batch.Y.to(params["device"])

        Y_pred_emb = model(X)
        Y_pred = model.topk_search(Y_pred_emb, topk)

        Y_true = Y.view(-1, 1).detach().cpu().numpy()
        Y_output = np.hstack([Y_pred, Y_true])
        print(Y_true)
        raise
        output_pred.append(Y_output)

    output_pred = np.concatenate(output_pred, axis=0)
    return output_pred

train_pred = predict(train_iter)
val_pred = predict(val_iter)
test_pred = predict(test_iter)

all_pred = np.concatenate([train_pred, val_pred, test_pred], axis=0)
np.savetxt(makedir(["all_preds"], "top-{}.csv".format(topk)), all_pred, delimiter=",")


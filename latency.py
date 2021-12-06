from model import RNN
import torch
import time
import json
import os
from data_loader import get_data_loaders
import pandas as pd

result_dir = "result/scratch/dropout_0"
data_path = "data/trace.txt"

with open(os.path.join(result_dir, "params.json"), "r") as f:
    params = json.load(f)

if torch.cuda.is_available():
    params["device"] = "cuda"
else:
    params["device"] = "cpu"

# load model
model_path = os.path.join(result_dir, "best_model", "acc-top10.model")
model = RNN(params).to(params["device"])

if params["device"] == "cuda":
    model.load_state_dict(torch.load(model_path))
else:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

model.index_embedding(use_gpu= (params["device"]!="cpu"))
# load data

train_iter, val_iter, test_iter, pid_to_idx = get_data_loaders(data_path, params)


for batch in test_iter:
    result = []
    for i in range(10):
        summary = []
        X = batch.X.to(params["device"])[0:1, i:i+1]
        Y = batch.Y.to(params["device"])[0:1, i:i+1]

        tic = time.time()
        Y_pred_emb = model(X)
        t1 = (time.time() - tic) / len(X)
        summary.append(t1)

        for topk in [1, 10, 20, 50]:
            tic = time.time()
            model.topk_search(Y_pred_emb, topk)
            t2 = (time.time() - tic) / len(X)
            summary.append(t2)

        result.append(summary)

    result = pd.DataFrame(result, columns=["inference", "search_top1", "search_top10", "search_top20", "search_top50"])

    result.to_csv("latency.csv", index=False)

result = []
for i, batch in enumerate(test_iter):
    if i > 10:
        break
        
    summary = []

    X = batch.X.to(params["device"])
    Y = batch.Y.to(params["device"])

    tic = time.time()
    Y_pred_emb = model(X)
    t1 = (time.time() - tic) / (X.shape[0] * X.shape[1])

    summary.append(t1)

    for topk in [1, 10, 20, 50]:
        tic = time.time()
        model.topk_search(Y_pred_emb, topk)
        t2 = (time.time() - tic) / (X.shape[0] * X.shape[1])
        summary.append(t2)

    result.append(summary)

result = pd.DataFrame(result, columns=["inference", "search_top1", "search_top10", "search_top20", "search_top50"])
result.to_csv("latency_batch.csv", index=False)

    

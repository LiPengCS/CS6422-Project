import json
import pandas as pd

result_dir = "pretrained_1028/dropout_0/metric_logging.json"

with open(result_dir, "r") as f:
    result = json.load(f)

items = ["acc", "acc_miss", "acc_hit"]

summary = []
for metric in items:
    summ = [metric]
    for k in [1, 10, 20, 50]:
        key = "test_top" + str(k)
        res_k = result[key]
        m = max(res_k[metric]["values"])
        summ.append(m)
    summary.append(summ)

summary = pd.DataFrame(summary, columns=["metric", "1", "10", "20", "50"])
summary.to_csv("summary.csv", index=False)

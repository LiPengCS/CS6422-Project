from tqdm import tqdm
from utils import SummaryWriter, AverageLoss, AverageAccuracy, AverageMetrics, makedir
import time
import numpy as np
from copy import deepcopy
import json
import torch

def train(model, train_iter, optimizer, params):
    model.train()
    device = params["device"]
    tr_loss = AverageLoss()

    for batch in train_iter:
        X = batch.X.to(device)
        Y = batch.Y.to(device)
        Y_pred_emb = model(X)
        loss = model.compute_loss(Y_pred_emb, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tr_loss.update(loss.item())

    return tr_loss()

def update_metrics(Y_pred, Y_true, is_miss, metrics):
    # reshape
    Y_true = Y_true.view(-1, 1).detach().cpu().numpy()
    is_miss = is_miss.view(-1,).detach().cpu().numpy()
    is_equal = (Y_pred == Y_true)

    for k in [1, 10, 20, 50]:
        is_correct = np.any(is_equal[:, :k], axis=1).astype(int)

        # accuracy
        metrics.add("acc-top{}".format(k), AverageAccuracy())
        metrics.update("acc-top{}".format(k), n_correct=sum(is_correct), n_example=len(is_correct))

        # miss accuracy
        miss_correct = is_correct * is_miss
        metrics.add("acc_miss-top{}".format(k), AverageAccuracy())
        metrics.update("acc_miss-top{}".format(k), n_correct=sum(miss_correct), n_example=sum(is_miss))

        # hit accuracy
        is_hit = (is_miss == 0).astype(int)
        hit_correct = is_correct * is_hit
        metrics.add("acc_hit-top{}".format(k), AverageAccuracy())
        metrics.update("acc_hit-top{}".format(k), n_correct=sum(hit_correct), n_example=sum(is_hit))

    return metrics

def evaluate(model, test_iter, params):
    model.eval()
    device = params["device"]

    test_loss = AverageLoss()
    metrics = AverageMetrics()

    model.index_embedding(use_gpu= (params["device"]!="cpu"))

    output_pred = []

    for batch in test_iter:
        X = batch.X.to(device)
        Y = batch.Y.to(device)
        Y_pred_emb = model(X)
        loss = model.compute_loss(Y_pred_emb, Y)

        Y_pred = model.topk_search(Y_pred_emb, params["top_k"])
        is_miss = batch.is_miss

        update_metrics(Y_pred, Y, is_miss, metrics)

        test_loss.update(loss.item())

        if params["output_pred"]:
            Y_true = Y.view(-1, 1).detach().cpu().numpy()
            Y_output = np.hstack([Y_pred, Y_true])
            output_pred.append(Y_output)

    if params["output_pred"]:
        output_pred = np.concatenate(output_pred, axis=0)

    return test_loss(), metrics(), output_pred

def train_evaluate(model, train_iter, val_iter, test_iter, optimizer, params, log_dir=None):
    if log_dir is not None:
        writer = SummaryWriter(log_dir)

    best_val_acc = float("-inf")
    val_acc = -1
    test_acc = -1
    p_bar = tqdm(range(params["num_epochs"]))

    best_metric = {}
    best_model = {}
    best_pred = {}

    for e in p_bar:
        tr_loss = train(model, train_iter, optimizer, params)

        if log_dir is not None:
            writer.add_scalar('tr_loss', tr_loss, global_step=e)

        if (e+1) % 10 == 0 or e == params["num_epochs"]-1:
            val_loss, val_metrics, val_pred = evaluate(model, val_iter, params)
            test_loss, test_metrics, test_pred = evaluate(model, test_iter, params)

            val_acc = val_metrics["acc-top10"]
            test_acc = test_metrics["acc-top10"]

            for name, value in test_metrics.items():
                if "acc-top" not in name:
                    continue

                if name not in best_metric:
                    best_metric[name] = {"test": value, "val": val_metrics[name]}
                    best_model[name] = deepcopy(model.state_dict())
                    best_pred[name] = {"val":val_pred, "test":test_pred}
                else:
                    if value > best_metric[name]["test"]:
                        best_metric[name] = {"test": value, "val": val_metrics[name]}
                        best_model[name] = deepcopy(model.state_dict())
                        best_pred[name] = {"val":val_pred, "test":test_pred}
 
            writer.add_scalar('val_loss', val_loss, global_step=e)
            writer.add_scalar('test_loss', test_loss, global_step=e)
            writer.add_metrics('val', val_metrics, global_step=e)
            writer.add_metrics('test', test_metrics, global_step=e)

        p_bar.set_postfix(tr_loss=tr_loss, val_acc=val_acc, test_acc=test_acc)

    writer.close()

    for name, metric_dict in best_metric.items():
        with open(makedir([log_dir, "best_result"], "{}.json".format(name)), "w") as f:
            json.dump(metric_dict, f, indent=4)

    for name, model_dict in best_model.items():
        torch.save(model_dict, makedir([log_dir, "best_model"], "{}.model".format(name)))

    if params["output_pred"]:
        for name, pred_dict in best_pred.items():
            np.savetxt(makedir([log_dir, "pred"], "{}_val_pred.csv".format(name)), pred_dict["val"], delimiter=",")
            np.savetxt(makedir([log_dir, "pred"], "{}_test_pred.csv".format(name)), pred_dict["test"], delimiter=",")

    return best_metric, best_model
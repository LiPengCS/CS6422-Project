import numpy as np
import pandas as pd
import torch
import os
import json
import shutil
from matplotlib import pyplot as plt
from collections import defaultdict

def makedir(dir_list, file=None, remove_old_dir=False):
    save_dir = os.path.join(*dir_list)

    if remove_old_dir and os.path.exists(save_dir) and file is None:
        shutil.rmtree(save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if file is not None:
        save_dir = os.path.join(save_dir, file)
    return save_dir

class SummaryWriter(object):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.scalar_logging = {}
        self.metric_logging = {}

    def logging(self, name, x, global_step, logging_dict):
        if name not in logging_dict:
            logging_dict[name] = {"indices":[global_step], "values":[x]}
        else:
            logging_dict[name]["indices"].append(global_step)
            logging_dict[name]["values"].append(x)

    def add_scalar(self, name, x, global_step):
        self.logging(name, x, global_step, self.scalar_logging)

    def add_metrics(self, prefix, metrics, global_step):
        for name, value in metrics.items():
            metric_name, param = name.split("-")
            key = prefix+ "_" + param
            if key not in self.metric_logging:
                self.metric_logging[key] = {}
            self.logging(metric_name, value, global_step, self.metric_logging[key])

    def plot_logging(self):
        save_dir = makedir([self.log_dir, "figures"])
        for name, scalar in self.scalar_logging.items():
            plt.plot(scalar["indices"], scalar["values"])
            plt.xlabel("epoch")
            plt.ylabel(name)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "{}.png".format(name)))
            plt.clf()

        for name, metric_values in self.metric_logging.items():
            for metric_name, scalar in metric_values.items():
                plt.plot(scalar["indices"], scalar["values"], label=metric_name)

            plt.xlabel("epoch")
            plt.ylabel("metrics")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "{}_metrics.png".format(name)))
            plt.clf()

    def save_logging(self):
        with open(makedir([self.log_dir, "logging"], "scalar_logging.json"), "w") as f:
            json.dump(self.scalar_logging, f, indent=4)

        with open(makedir([self.log_dir, "logging"], "metric_logging.json"), "w") as f:
            json.dump(self.metric_logging, f, indent=4)

    def close(self):
        self.save_logging()
        self.plot_logging()

class AverageAccuracy(object):
    def __init__(self):
        self.n_examples = 0
        self.n_correct = 0

    def update(self, n_correct, n_example):
        self.n_examples += n_example
        self.n_correct += n_correct

    def __call__(self):
        return self.n_correct / self.n_examples

class AverageLoss(object):
    def __init__(self):
        self.loss = 0
        self.n_batches = 0

    def update(self, loss):
        """

        Parameters
        ----------
        is_correct (np.array)
            A list of booleans (0 or 1) indicating whether prediction is correct or not
        """
        self.loss += loss
        self.n_batches += 1

    def __call__(self):
        return self.loss / self.n_batches

class AverageMetrics(object):
    def __init__(self):
        self.metric_objects = {}

    def add(self, name, obj):
        if name not in self.metric_objects:
            self.metric_objects[name] = obj

    def update(self, name, **kwargs):
        self.metric_objects[name].update(**kwargs)

    def __call__(self):
        metrics = {}
        for name, obj in self.metric_objects.items():
            metrics[name] = obj()
        return metrics
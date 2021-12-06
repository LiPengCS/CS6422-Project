import faiss
import torch
import torch.nn as nn
import time
import numpy as np

class RNN(nn.Module):
    """ Prediction of Next word based on the MAX_SEQ_LEN Sequence """
    def __init__(self, params, pretrained_embedding=None):
        super(RNN, self).__init__()
        self.hidden_dim = params["hidden_dim"]
        self.embedding_dim = params["embedding_dim"]
        self.vocab_size = params["vocab_size"]

        if pretrained_embedding is not None:
            weights = torch.FloatTensor(pretrained_embedding.vectors)
            self.embedding = nn.Embedding.from_pretrained(weights)
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.rnn = nn.GRU(self.embedding_dim, self.hidden_dim, params["num_layers"], dropout=params["dropout"])
        self.decoder = nn.Linear(self.hidden_dim, self.embedding_dim)
        self.loss_fn = nn.MSELoss()

    def forward(self, X):
        embed = self.embedding(X)
        out, hidden = self.rnn(embed)
        Y_pred_emb = self.decoder(out)
        return Y_pred_emb

    def compute_loss(self, Y_pred_emb, Y_true):
        Y_true_emb = self.embedding(Y_true)
        loss = self.loss_fn(Y_pred_emb, Y_true_emb)
        return loss

    def index_embedding(self, use_gpu=False):
        self.embed_index = faiss.IndexFlatL2(self.embedding_dim)
        if use_gpu:
            self.embed_index = faiss.index_cpu_to_all_gpus(self.embed_index)

        embed_vectors = self.embedding.weight.detach().cpu().numpy()
        self.embed_index.add(embed_vectors)

    def topk_search(self, Y_pred_emb, top_k=5):
        query = Y_pred_emb.view(-1, self.embedding_dim).detach().cpu().numpy()
        _, topk_idx = self.embed_index.search(x=query, k=top_k)
        return topk_idx

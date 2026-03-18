import torch
import torch.nn as nn
from torch import Tensor

import math


def scaled_dot_product_attention(query:Tensor, key:Tensor, value:Tensor, mask=None):

    """
    query: [batch_size, n_heads, seq_len_q, d_k]
    key:   [batch_size, n_heads, seq_len_k, d_k]
    value: [batch_size, n_heads, seq_len_v, d_k]
    mask:  [batch_size, 1, seq_len_q, seq_len_k] 或兼容形状
    """
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # scores: [batch_size, n_heads, seq_len_q, seq_len_k]

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)

    return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x:Tensor):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        x = x.transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]
        return x

    def combine_heads(self, x:Tensor):
        # x: [batch_size, n_heads, seq_len, d_k]
        batch_size, n_heads, seq_len, d_k = x.size()
        x = x.transpose(1, 2).contiguous()  # [batch_size, seq_len, n_heads, d_k]
        x = x.view(batch_size, seq_len, self.d_model)
        return x

    def forward(self, query, key, value, mask=None):
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        attn_output = self.combine_heads(attn_output)
        output = self.fc_out(attn_output)

        return output, attn_weights
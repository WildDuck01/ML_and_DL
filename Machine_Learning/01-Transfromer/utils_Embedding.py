import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchtyping import TensorType, patch_typeguard # 用于tensor shape级类型提示
from typeguard import typechecked
patch_typeguard()


class TokenEmbedding(nn.Module):
    '''
        1.1 word Embedding
        重点利用 torch.nn.Embedding 将离散的整数索引（如单词 ID、类别 ID）映射为连续的低维向量（目前忽略其中间实现）
        我们人工给定一个词汇表大小为vocab_size，其每个索引映射到一个 预定义的dim_model 维的向量
    '''
    def __init__(self, vocab_size:int, dim_model:int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=dim_model)
        
    @typechecked
    def forward(self, x:TensorType["batch","seq"]) -> TensorType["batch","seq","dim"]:
        return self.embedding(x)  # [batch_size, seq_len, 预定义的embedding维度]
    
    
class PositionalEncoding(nn.Module): # 以下为正弦位置编码实现
    '''
        1.2 position Embedding
        给输入的每个位置（比如句子里的每个单词）加一个 “位置标签” 让模型知道单词的先后顺序
        用正弦 / 余弦函数生成position 既保证位置越近标签越相似 又能支持超长句子
    '''
    def __init__(self, dim_model:int, max_len:int=5000, dropout:float=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, dim_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * 
                             (-math.log(10000.0) / dim_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维

        pe = pe.unsqueeze(0)  # [1, max_len, d_model] 这里还有细节 就是 pe 是一个固定的位置编码表 不参与训练的固定编码
        self.register_buffer("pe", pe) # 不针对该参数进行训练 这里有细节

    def forward(self, x:TensorType["batch","seq","dim"]) -> TensorType["batch","seq","dim"]:
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)
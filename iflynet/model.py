import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np



# 定义模型和其他组件
class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, dropout=0.5):#加大dropout层可以减小过拟合的概率
        super(Transformer, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = x.squeeze(0)  # 从三维张量中去除额外的维度
        x = self.embed(x)
        x = x.unsqueeze(0)  # 将嵌入后的张量恢复为三维张量
        x = self.positional_encoding(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.fc(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=700):#######################################################修改最大句子长度
        super(PositionalEncoding, self).__init__()

        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))

        self.positional_encoding = nn.Parameter(torch.zeros(max_seq_len, embed_dim))
        self.positional_encoding.data[:, 0::2] = torch.sin(position * div_term)
        self.positional_encoding.data[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        x = x + self.positional_encoding[:x.size(1), :]
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout):
        super(TransformerBlock, self).__init__()

        self.attention = MultiheadAttention(embed_dim, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x
        x = self.attention(x)
        x = self.dropout1(x)
        x = self.norm1(residual + x)
        residual = x
        x = self.fc(x)
        x = self.dropout2(x)
        x = self.norm2(residual + x)
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)

        x = torch.matmul(attention_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        #x = self.fc(x)
        x =self.fc(x.view(-1, self.embed_dim))
        return x
import torch
from torch import nn

from src.multi_head import MultiHeadAttention
from src.feed_forward import FeedForwardNetwork
from src.add_norm import AddNorm

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.add_norm1 = AddNorm(d_model)
        
        self.feed_forward = FeedForwardNetwork(d_model, d_ff)
        self.add_norm2 = AddNorm(d_model)
    
    def forward(self, x, mask=None):
        attention_output = self.self_attention(x, x, x, mask)
        x = self.add_norm1(x, attention_output)

        ffn_output = self.feed_forward(x)
        x = self.add_norm2(ffn_output)

        return x
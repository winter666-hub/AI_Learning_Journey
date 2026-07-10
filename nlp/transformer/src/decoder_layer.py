import torch
from torch import nn
from src.multi_head import MultiHeadAttention
from src.add_norm import AddNorm
from src.feed_forward import FeedForwardNetwork

class DecoderLayer(nn.Module):
    def __init__(self,
                 d_model=512,
                 num_heads=8,
                 d_ff=2048):
        super().__init__()

        # Decoder 내부의 Self Attention
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.add_norm1 = AddNorm(d_model)

        # Encoder와 Decoder를 연결하는 Cross Attention
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.add_norm2 = AddNorm(d_model)

        self.feed_forward = FeedForwardNetwork(d_model, d_ff)
        self.add_norm3= AddNorm(d_model)

    def forward(self, x, encoder_output):
        # 1. Decoder 내부 Self-Attention
        attention_output = self.self_attention(x, x, x)
        x = self.add_norm1(x, attention_output)

        # 2. Encoder의 출력 정보를 참고하는 Cross Attention
        cross_output = self.cross_attention(
            x,              # Decoder가 현재까지 생성한 정보 (Q)
            encoder_output, # Encoder가 이해한 원문 정보 (K)
            encoder_output  # Encoder가 전달할 정보 (V) 
        )
        x = self.add_norm2(x, cross_output)
        
        ffn_output = self.feed_forward(x)
        x = self.add_norm3(x, ffn_output)
        
        return x
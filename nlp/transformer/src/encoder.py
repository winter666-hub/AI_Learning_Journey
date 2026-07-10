import torch
from torch import nn

from src.encoder_layer import EncoderLayer

class Encoder(nn.Module):
    def __init__(self,
                 num_layers=6,
                 d_model=512,
                 num_heads=8,
                 d_ff=2048):
        super().__init__()

        #EncoderLayer를 num_layer개 쌓는다
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        # 각 EncoderLayer를 순서대로 통과
        for layer in self.layers:
            x = layer(x)

        return x
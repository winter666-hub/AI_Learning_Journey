from torch import nn
from src.decoder_layer import DecoderLayer

class Decoder(nn.Module):
    def __init__(self,
                 num_layers=6,
                 d_model=512,
                 num_heads=8,
                 d_ff=2048
                 ):
        super().__init__()
        
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff)
             for _ in range(num_layers)
            ]
        )

    def forward(self, x, encoder_output):
        for layer in self.layers:
            # x : 이전 DecoderLayer의 출력
            # encoder_output : Encoder가 이해한 원문 정보
            x = layer(x, encoder_output)
            
        return x
import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model=512):
        super().__init__()

        # Positional Encoding을 저장할 행렬
        pe = torch.zeros(max_len, d_model)

        # 각 토큰의 위치 (0 ~ max_len-1)
        # shape: (max_len,) -> (max_len, 1)
        position = torch.arange(max_len).unsqueeze(1)
        
        # 각 차원마다 다른 변화 속도를 만들기 위한 값
        # 짝수 차원(0, 2, 4, ...)에 해당하는 스케일을 계산
        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            (-math.log(10000.0) / d_model)    
        )
        # 짝수 차원에는 sin 함수 적용
        pe[:, 0::2] = torch.sin(position * div_term)
        # 홀수 차원에서 cos 함수 적용
        pe[:, 1::2] = torch.cos(position * div_term)

        # Positional Encoding은 학습되지 않는 값으므로 buffer로 등록
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # 현재 입력 문장의 길이(seq_len)에 해당하는 Positional Encoding만 선택
        # Embedding 벡터에 위치 정보를 더함
        x = x + self.pe[:, :x.size(1)]
        return x
from torch import nn
from positional_encoding import PositionalEncoding
from src.encoder import Encoder
from src.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model=512,
                 num_heads=8,
                 d_ff=2048,
                 num_layers=6):
        super().__init__()

        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model)
        
        self.encoder = Encoder(num_layers,
                               d_model,
                               num_heads,
                               d_ff)
        
        self.decoder = Decoder(num_layers,
                               d_model,
                               num_heads,
                               d_ff)
        
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        # Encoder
        # 토큰 번호 -> 임베딩 벡터
        src = self.encoder_embedding(src)
        # 위치 정보 추가
        src = self.positional_encoding(src)
        # 입력 문장을 이해한 결과
        encoder_output = self.encoder(src)

        # Decoder
        # 토큰 번호 -> 임베딩 벡터
        tgt = self.decoder_embedding(tgt)
        # 위치 정보 추가
        tgt = self.positional_encoding(tgt)
        # Encoder의 정보를 참고하여 다음 단어 생성
        decoder_output = self.decoder(tgt, encoder_output)

        # 단어별 점수(logits) 계산
        output = self.fc(decoder_output)

        return output
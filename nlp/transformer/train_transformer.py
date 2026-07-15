import torch
from torch import nn
from torch.optim import Adam

from src.transformer import Transformer

# 1. 모델 설정

# 입력 문장(src)의 단어 개수
src_vocab_size = 1000
# 출력 문장(tgt)의 단어 개수
tgt_vocab_size = 1000

# Transformer 생성
model = Transformer(
    src_vocab_size,
    tgt_vocab_size
)

# 2. Loss, Optimizer

# 실제 정답 token id를 비교하는 Loss
criterion = nn.CrossEntropyLoss()

# Loss를 기반으로 모델의 weight를 수정하는 optimizer
optimizer = Adam(model.parameters(),
                 lr=0.0001)

# 3. 입력 데이터 생성

batch_size = 2

# Encoder에 들어가는 원문 길이
src_seq_len = 5

# Decoder 입력 길이
tgt_seq_len = 4

# Encoder 입력
# (batch_size, source sequence length)
src = torch.randint(
    0,
    src_vocab_size,
    (batch_size, src_seq_len)
)

# Decoder 입력
# 실제 학습에서는 <BOS>가 붙은 정답 문장
tgt_input = torch.randint(
    0,
    tgt_vocab_size,
    (batch_size, tgt_seq_len)
)

# 정답 Label
# 실제 학습에서는 한 칸 이동한 정답
target = torch.randint(
    0,
    tgt_vocab_size,
    (batch_size, tgt_seq_len)
)

print(src.shape)
print(tgt_input.shape)
print(target.shape)

# 4. Forward

# Encoder와 Decoder를 통과하여
# 각 위치별 단어 점수(logits)를 출력
output = model(
    src,
    tgt_input
)

print(output.shape)

# 5. Loss 계산

# 현재 (2, 4, 1000) CrossEntropyLoss는 (예측 개수, vocab_size)
# (8, 1000)형태로 변경
output = output.reshape(
    -1,
    tgt_vocab_size
)

# (2, 4) -> (8,)
target = target.reshape(-1)

loss = criterion(
    output,
    target
)

print(loss.item())


# 6. Backpropagation

# 매 학습 step마다 초기화 필요
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("training step complete")
import torch
import pickle
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.tokenizer import Tokenizer
from src.dataset import TranslationDataset
from src.transformer import Transformer

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")

# 1. 예제 번역 데이터
src_sentences = [
    "나는 학생이다",
    "나는 개발자다"
]

tgt_sentences = [
    "I am student",
    "I am developer"
]

# 2. Tokenizer
tokenizer = Tokenizer()

# Vocabulary 생성
tokenizer.build_vocab(
    src_sentences + tgt_sentences
)

# Vocabulary 크기
vocab_size = len(tokenizer.word_to_id)

# 3. Dataset / DataLoader

dataset = TranslationDataset(
    src_sentences,
    tgt_sentences,
    tokenizer,
    max_length=6
)

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

# 4. Transformer 생성

model = Transformer(
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size
).to(device)

# 5. Loss / Optimizer
criterion = nn.CrossEntropyLoss()

optimizer = Adam(
    model.parameters(),
    lr=0.0001
)

# 6. Training
epochs = 100

for epoch in range(epochs):
    for batch in dataloader:

        src = batch["src"].to(device)
        tgt_input = batch["tgt_input"].to(device)
        target = batch["target"].to(device)

        # Forward
        output = model(src, tgt_input)

        output = output.reshape(-1, vocab_size)
        target = target.reshape(-1)

        # Loss
        loss = criterion(output, target)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss : {loss.item():.4f}"
    )

print("training complete")

torch.save(
    model.state_dict(),
    "models/transformer_model.pth"
)

print("model saved")

# Tokenizer 저장
with open(
    "models/tokenizer.pkl",
    # 쓰기 모드
    "wb"
) as f:
    pickle.dump(
        tokenizer,
        f
    )

print("tokenizer saved")
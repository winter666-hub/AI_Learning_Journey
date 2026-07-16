import torch
from torch.utils.data import DataLoader

from src.tokenizer import Tokenizer
from src.dataset import TranslationDataset

# 예제 데이터
src_sentences = [
    "나는 학생이다",
    "나는 개발자다"
]

tgt_sentences = [
    "I am student",
    "I am developer"
]

tokenizer = Tokenizer()

tokenizer.build_vocab(
    src_sentences + tgt_sentences
)

dataset = TranslationDataset(
    src_sentences,
    tgt_sentences,
    tokenizer,
    max_length=6
)

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False
)

for batch in dataloader:
    print("src")
    print(batch["src"])

    print()

    print("tgt_input")
    print(batch["tgt_input"])

    print()

    print("target")
    print(batch["target"])
import torch
from src.transformer import Transformer

model = Transformer(
    src_vocab_size=1000,
    tgt_vocab_size=1000
)

src = torch.randint(0, 1000, (2, 5))
tgt = torch.randint(0, 1000, (2, 4))

output = model(src, tgt)

print(output.shape)
print("src shape:", src.shape)
print("tgt shape:", tgt.shape)

output = model(src, tgt)

print("output shape:", output.shape)
import torch
import pickle

from src.transformer import Transformer

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")

vocap_size = 10

model = Transformer(
    src_vocab_size=vocap_size,
    tgt_vocab_size=vocap_size
).to(device)

model.load_state_dict(
    torch.load(
        "models/transformer_model.pth",
        map_location=device
    )
)

model.eval()

print("model loaded")

with open(
    "models/tokenizer.pkl",
    # 읽기 모드
    "rb"
) as f:
    tokenizer = pickle.load(f)

print("tokenizer loaded")

sentence = "나는 학생이다"
max_length = 6

src_ids = tokenizer.encode(sentence)
src_ids = tokenizer.pad_sequence(src_ids, max_length)

src = torch.tensor(
    [src_ids],
    device=device
)

decoder_input = torch.tensor(
    [[tokenizer.word_to_id["<BOS>"]]],
    device=device
)

with torch.no_grad():
    for _ in range(max_length):
        output = model(
            src,
            decoder_input
        )

        last_logits = output[:, -1, :]

        next_token = torch.argmax(
            last_logits,
            dim=-1
        )

        decoder_input = torch.cat(
            [
                decoder_input,
                next_token.unsqueeze(1)
            ], dim=-1
        )

        if next_token.item() == tokenizer.word_to_id["<EOS>"]:
            break

result = tokenizer.decode(
    decoder_input.squeeze(0).tolist()
)

print(result)
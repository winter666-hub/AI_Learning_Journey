from src.tokenizer import Tokenizer

# 학습에 사용할 문장 데이터
sentences = [
    "나는 학생이다",
    "나는 개발자다",
    "나는 공부한다"
]

# tokenizer 생성
tokenizer = Tokenizer()

# 문장을 보고 vocabulary 생성
tokenizer.build_vocab(sentences)

# 단어 -> 숫자 확인
print("word_to_id")
print(tokenizer.word_to_id)

# 숫자 -> 단어 확인
print("\nid_to_word")
print(tokenizer.id_to_word)

# 문장을 숫자로 변환
encoded = tokenizer.encode(
    "나는 학생이다"
)

print("\nencoded")
print(encoded)

padded = tokenizer.pad_sequence(
    encoded,
    max_length=6
)

print("padded")
print(padded)

# 숫자를 다시 문장으로 변환
decoded = tokenizer.decode(
    encoded
)

print("\ndecoded")
print(decoded)


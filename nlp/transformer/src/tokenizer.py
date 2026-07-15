class Tokenizer:
    def __init__(self):
        # 특수 토큰
        self.special_tokens = {
            "<PAD>": 0,
            "<BOS>": 1,
            "<EOS>": 2,
        }

        # 단어 -> 숫자
        self.word_to_id = {}

        # 숫자 -> 단어
        self.id_to_word = {}

    def build_vocab(self, sentences):
        self.word_to_id = self.special_tokens.copy()

        for sentence in sentences:
            tokens = sentence.split()
            for token in tokens:
                if token not in self.word_to_id:
                    self.word_to_id[token] = len(self.word_to_id)
        
        # 역방향 dictionary 생성
        self.id_to_word = {
            id: word
            for word, id in self.word_to_id.items()
        }

    def encode(self, sentence, add_special_tokens=True):
        # 문장 -> token id
        tokens = sentence.split()

        ids = []

        # 문장 시작 토큰 추가
        if add_special_tokens:
            ids.append(
                self.word_to_id["<BOS>"]
            )

        # 일반 단어 변환
        for token in tokens:
            ids.append(
                self.word_to_id[token]
            )
        
        # 문장 종료 토큰 추가
        if add_special_tokens:
            ids.append(
                self.word_to_id["<EOS>"]
            )

        return ids
    
    def decode(self, ids):
        tokens = []

        for id in ids:
            token = self.id_to_word[id]

            if token not in ["<BOS>", "<EOS>", "<PAD>"]:
                tokens.append(token)

        return " ".join(tokens)
    
    def pad_sequence(self, ids, max_length):
        pad_id = self.word_to_id["<PAD>"]

        padded_ids = ids.copy()

        # 부족한 길이만큼 PAD 추가
        while len(padded_ids) < max_length:
            padded_ids.append(pad_id)

        return padded_ids[:max_length]
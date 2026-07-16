from torch.utils.data import Dataset

class TranslationDataset(Dataset):

    def __init__(
        self,
        src_sentences,
        tgt_sentences,
        tokenizer,
        max_length=10
    ):  
        # 원문(Encoder 입력)
        self.src_sentences = src_sentences
        # 번역문(Decoder 입력 및 정답)
        self.tgt_sentences = tgt_sentences

        # 문장을 Token ID로 변환하는 Tokenizer
        self.tokenizer = tokenizer
        # 모든 문장의 길이를 맞추기 위한 최대 길이
        self.max_length = max_length


    def __len__(self):
        # 데이터셋에 들어있는 문장의 개수 반환
        return len(self.src_sentences)


    def __getitem__(self, index):

        # index번째 원문과 번역문 가져오기
        src_sentence = self.src_sentences[index]
        tgt_sentence = self.tgt_sentences[index]

        # 문장을 Token ID로 변환       
        src_ids = self.tokenizer.encode(
            src_sentence
        )
        tgt_ids = self.tokenizer.encode(
            tgt_sentence
        )

        # 길이가 다른 문장들을 같은 길이로 맞춤
        # 부족한 부분은 PAD 토큰 추가
        src_ids = self.tokenizer.pad_sequence(
            src_ids,
            self.max_length
        )

        tgt_ids = self.tokenizer.pad_sequence(
            tgt_ids,
            self.max_length
        )


        # Decoder 입력
        # 마지막(EOS)을 제거
        tgt_input = tgt_ids[:-1]

        # 모델이 맞춰야 하는 정답
        # 첫 번째(BOS)를 제거
        target = tgt_ids[1:]

        # Transformer 학습에 필요한 데이터 반환
        return {
            "src": src_ids,
            "tgt_input": tgt_input,
            "target": target
        }
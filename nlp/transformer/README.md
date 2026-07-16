# Transformer from Scratch

**Attention Is All You Need (2017)** 논문를 기반으로 Transformer를 처음부터 직접 구현한 프로젝트입니다.

PyTorch의 `nn.Transformer`를 사용하지 않고, Transformer의 핵심 구성 요소를 하나씩 직접 구현하며 논문의 구조와 동작 원리를 이해하는 것을 목표로 했습니다.

---

## Learning Goals

- Transformer의 전체 구조 이해
- Self-Attention과 Multi-Head Attention 구현
- Encoder와 Decoder의 동작 원리 이해
- Teacher Forcing을 이용한 학습 구현
- Greedy Decoding을 이용한 추론 구현
- 논문의 수식과 실제 코드의 연결 이해

---

## Implemented Components

- Token Embedding
- Positional Encoding
- Scaled Dot-Product Attention
- Multi-Head Attention
- Feed Forward Network
- Add & Layer Normalization
- Encoder
- Decoder
- Transformer
- Tokenizer
- Translation Dataset
- Training Pipeline
- Model Save & Load
- Greedy Decoding Inference

---

## Project Structure

```
transformer/
├── models/
│   ├── transformer_model.pth
│   └── tokenizer.pkl
├── src/
│   ├── add_norm.py
│   ├── attention.py
│   ├── dataset.py
│   ├── decoder.py
│   ├── embedding.py
│   ├── encoder.py
│   ├── feed_forward.py
│   ├── multi_head.py
│   ├── tokenizer.py
│   └── transformer.py
├── train_transformer.py
├── inference.py
├── test_dataloader.py
├── test_tokenizer.py
└── README.md
```

---

## Example

**Input**

```
나는 학생이다
```

**Output**

```
I am student
```

---

## Reference

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.

## 배운 점

- Transformer의 전체 동작 과정을 직접 구현하며 구조를 이해했다.
- 논문의 수식이 실제 PyTorch 코드에서 어떻게 구현되는지 익혔다.
- Teacher Forcing을 이용한 학습 과정을 구현했다.
- Greedy Decoding을 구현하며 Auto-Regressive 추론 과정을 이해했다.
- Tokenizer부터 모델 저장 및 추론까지 전체 파이프라인을 구현했다.
import torch
import math
import torch.nn.functional as F
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)

    K_t = K.transpose(-2, -1)

    scores = torch.matmul(Q, K_t)
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask==0, float('-inf'))
    
    attn_weights = F.softmax(scores, dim=-1)

    output = torch.matmul(attn_weights, V)

    return scores, attn_weights, output

def generate_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len))

print("attention.py loaded")
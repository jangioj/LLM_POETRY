import torch
import torch.nn as nn

from data import vocab_size, CONTEXT_LENGTH

N_EMBD = 256
N_HEAD = 8
N_LAYER = 6
DROPOUT = 0.1

import torch
import torch.nn as nn

N_EMBD = 256
DROPOUT = 0.1

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.net(x)

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

        assert N_EMBD % N_HEAD == 0, "N_EMBD must be divisible by N_HEAD"

        self.head_size = N_EMBD // N_HEAD

        self.key = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.query = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.value = nn.Linear(N_EMBD, N_EMBD, bias=False)

        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)    # (B, T, C)
        q = self.query(x)  # (B, T, C)
        v = self.value(x)  # (B, T, C)

        # split into heads
        k = k.view(B, T, N_HEAD, self.head_size).transpose(1, 2)  # (B, N_HEAD, T, head_size)
        q = q.view(B, T, N_HEAD, self.head_size).transpose(1, 2)  # (B, N_HEAD, T, head_size)
        v = v.view(B, T, N_HEAD, self.head_size).transpose(1, 2)  # (B, N_HEAD, T, head_size)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=DROPOUT if self.training else 0.0,
            is_causal=True
        )  # (B, N_HEAD, T, head_size)

        # merge heads back
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        out = self.proj(out)
        out = self.dropout(out)

        return out

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()

        self.ln2 = nn.LayerNorm(N_EMBD)
        self.ffwd = FeedForward()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class CharGPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(CONTEXT_LENGTH, N_EMBD)

        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)                      # (B, T, C)
        pos = torch.arange(T, device=idx.device)                      # (T,)
        pos_emb = self.position_embedding_table(pos)                  # (T, C)

        x = tok_emb + pos_emb                                         # (B, T, C)
        x = self.blocks(x)                                            # (B, T, C)
        x = self.ln_f(x)                                              # (B, T, C)
        logits = self.lm_head(x)                                      # (B, T, vocab_size)

        return logits

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -CONTEXT_LENGTH:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]  # take last time step
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)

        return idx
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import get_batch, CONTEXT_LENGTH
from model import vocab_size, N_EMBD, N_HEAD, N_LAYER, DROPOUT


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttentionBenchmark(nn.Module):
    def __init__(self, implementation: str):
        super().__init__()
        assert implementation in {"sdpa", "manual"}
        assert N_EMBD % N_HEAD == 0

        self.implementation = implementation
        self.head_size = N_EMBD // N_HEAD

        self.key = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.query = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.value = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

        # causal mask for the manual implementation
        mask = torch.tril(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH, dtype=torch.bool))
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        k = k.view(B, T, N_HEAD, self.head_size).transpose(1, 2)
        q = q.view(B, T, N_HEAD, self.head_size).transpose(1, 2)
        v = v.view(B, T, N_HEAD, self.head_size).transpose(1, 2)

        if self.implementation == "sdpa":
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=DROPOUT if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Traditional masked attention:
            # scores: (B, N_HEAD, T, T)
            scores = q @ k.transpose(-2, -1)
            scores = scores / (self.head_size ** 0.5)

            mask = self.causal_mask[:T, :T]
            scores = scores.masked_fill(~mask, float("-inf"))

            weights = F.softmax(scores, dim=-1)
            weights = F.dropout(weights, p=DROPOUT, training=self.training)

            out = weights @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class TransformerBlockBenchmark(nn.Module):
    def __init__(self, implementation: str):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = MultiHeadAttentionBenchmark(implementation)
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.ffwd = FeedForward()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class CharGPTBenchmark(nn.Module):
    def __init__(self, implementation: str):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(CONTEXT_LENGTH, N_EMBD)
        self.blocks = nn.Sequential(
            *[TransformerBlockBenchmark(implementation) for _ in range(N_LAYER)]
        )
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_emb = self.token_embedding_table(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos)

        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss


def benchmark_one(implementation, device, warmup_steps, benchmark_steps, lr):
    torch.manual_seed(42)

    model = CharGPTBenchmark(implementation).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Warm-up: important on GPU because first iterations include setup overhead.
    for _ in range(warmup_steps):
        xb, yb = get_batch("train")
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(xb, yb)
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times = []
    losses = []

    for _ in range(benchmark_steps):
        xb, yb = get_batch("train")
        xb, yb = xb.to(device), yb.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(xb, yb)
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

        times.append(end - start)
        losses.append(loss.item())

    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2

    return {
        "implementation": implementation,
        "mean_step_time_sec": sum(times) / len(times),
        "tokens_per_step": get_batch("train")[0].numel(),
        "mean_loss": sum(losses) / len(losses),
        "peak_memory_mb": peak_memory_mb,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--benchmark_steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--output_path", type=str, default="attention_benchmark.txt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    results = []
    for implementation in ["sdpa", "manual"]:
        print(f"\nBenchmarking: {implementation}")
        result = benchmark_one(
            implementation=implementation,
            device=device,
            warmup_steps=args.warmup_steps,
            benchmark_steps=args.benchmark_steps,
            lr=args.learning_rate,
        )
        results.append(result)
        print(result)

    lines = []
    lines.append("ATTENTION IMPLEMENTATION BENCHMARK")
    lines.append(f"device: {device}")
    if device.type == "cuda":
        lines.append(f"gpu: {torch.cuda.get_device_name(0)}")
    lines.append(f"context_length: {CONTEXT_LENGTH}")
    lines.append(f"n_layer: {N_LAYER}")
    lines.append(f"n_embd: {N_EMBD}")
    lines.append(f"n_head: {N_HEAD}")
    lines.append(f"dropout: {DROPOUT}")
    lines.append(f"warmup_steps: {args.warmup_steps}")
    lines.append(f"benchmark_steps: {args.benchmark_steps}")
    lines.append("")

    for r in results:
        lines.append(f"implementation: {r['implementation']}")
        lines.append(f"mean_step_time_sec: {r['mean_step_time_sec']:.6f}")
        lines.append(f"tokens_per_step: {r['tokens_per_step']}")
        lines.append(f"mean_loss: {r['mean_loss']:.6f}")
        if r["peak_memory_mb"] is not None:
            lines.append(f"peak_memory_mb: {r['peak_memory_mb']:.2f}")
        lines.append("")

    sdpa = next(r for r in results if r["implementation"] == "sdpa")
    manual = next(r for r in results if r["implementation"] == "manual")

    speedup = manual["mean_step_time_sec"] / sdpa["mean_step_time_sec"]
    lines.append(f"speedup_manual_over_sdpa: {speedup:.3f}x")

    if sdpa["peak_memory_mb"] is not None and manual["peak_memory_mb"] is not None:
        memory_ratio = manual["peak_memory_mb"] / sdpa["peak_memory_mb"]
        lines.append(f"memory_ratio_manual_over_sdpa: {memory_ratio:.3f}x")

    output_path = Path(args.output_path)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"\nSaved benchmark to: {output_path}")


if __name__ == "__main__":
    main()

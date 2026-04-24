import csv
import math
from datetime import datetime
from pathlib import Path
import json
import argparse
import time

import torch
import torch.nn as nn

from data import (
    get_batch,
    train_text,
    val_text,
    test_text,
    vocab_size,
    CONTEXT_LENGTH,
    BATCH_SIZE,
    encode,
    decode,
)
from model import CharGPT, N_EMBD, N_HEAD, N_LAYER, DROPOUT


device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# EXPERIMENT CONFIG
# =========================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--eval_interval", type=int, default=20)
    parser.add_argument("--eval_iters", type=int, default=10)
    parser.add_argument("--sample_every", type=int, default=100)
    parser.add_argument("--sample_tokens", type=int, default=400)
    parser.add_argument("--prompt", type=str, default="Mio padre era ")
    parser.add_argument("--checkpoint_every", type=int, default=100)

    return parser.parse_args()


args = parse_args()

LEARNING_RATE = args.learning_rate
NUM_STEPS = args.num_steps
EVAL_INTERVAL = args.eval_interval
EVAL_ITERS = args.eval_iters
SAMPLE_EVERY = args.sample_every
SAMPLE_TOKENS = args.sample_tokens
PROMPT = args.prompt
CHECKPOINT_EVERY = args.checkpoint_every
SAVE_BEST_MODEL = True

# =========================
# RUN SETUP
# =========================
run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
run_dir = Path("outputs") / run_name
run_dir.mkdir(parents=True, exist_ok=True)

checkpoints_dir = run_dir / "checkpoints"
checkpoints_dir.mkdir(parents=True, exist_ok=True)

log_file = run_dir / "run_log.txt"
metrics_file = run_dir / "metrics.csv"
config_file = run_dir / "config.txt"
final_model_file = run_dir / "final_model.pt"
best_model_file = run_dir / "best_model.pt"
config_json_file = run_dir / "config.json"


#CORE RUN
model = CharGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

#UTILITIES
def compute_loss(xb, yb):
    logits = model(xb)

    B, T, C = logits.shape
    logits = logits.view(B * T, C)
    targets = yb.view(B * T)

    loss = loss_fn(logits, targets)
    return loss


def write_log(text):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def write_config():
    config = {
        "run_name": run_name,
        "device": device,
        "train_length": len(train_text),
        "val_length": len(val_text),
        "test_length": len(test_text),
        "vocab_size": vocab_size,
        "context_length": CONTEXT_LENGTH,
        "batch_size": BATCH_SIZE,
        "n_embd": N_EMBD,
        "n_head": N_HEAD,
        "n_layer": N_LAYER,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
        "num_steps": NUM_STEPS,
        "eval_interval": EVAL_INTERVAL,
        "eval_iters": EVAL_ITERS,
        "sample_every": SAMPLE_EVERY,
        "sample_tokens": SAMPLE_TOKENS,
        "checkpoint_every": CHECKPOINT_EVERY,
        "save_best_model": SAVE_BEST_MODEL,
        "prompt": PROMPT,
    }

    with open(config_file, "w", encoding="utf-8") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    with open(config_json_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


def save_checkpoint(path: Path, step: int, best_val_loss: float):
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "config": {
                "learning_rate": LEARNING_RATE,
                "num_steps": NUM_STEPS,
                "eval_interval": EVAL_INTERVAL,
                "eval_iters": EVAL_ITERS,
                "sample_every": SAMPLE_EVERY,
                "sample_tokens": SAMPLE_TOKENS,
                "checkpoint_every": CHECKPOINT_EVERY,
                "context_length": CONTEXT_LENGTH,
                "batch_size": BATCH_SIZE,
                "n_embd": N_EMBD,
                "n_head": N_HEAD,
                "n_layer": N_LAYER,
                "dropout": DROPOUT,
                "prompt": PROMPT,
            },
        },
        path,
    )

@torch.no_grad()
def estimate_loss():
    losses = {}
    bpcs = {}

    model.eval()

    for split in ["train", "val"]:
        split_losses = []

        for _ in range(EVAL_ITERS):
            xb, yb = get_batch(split)
            xb, yb = xb.to(device), yb.to(device)

            loss = compute_loss(xb, yb)
            split_losses.append(loss.item())

        mean_loss = sum(split_losses) / len(split_losses)
        losses[split] = mean_loss
        bpcs[split] = mean_loss / math.log(2)

    model.train()
    return losses, bpcs

@torch.no_grad()
def generate_sample(prompt=PROMPT, max_new_tokens=SAMPLE_TOKENS):
    model.eval()

    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=max_new_tokens)
    text = decode(generated[0].tolist())

    model.train()
    return text

with open(metrics_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "train_loss", "val_loss", "train_bpc", "val_bpc"])

write_config()

write_log(f"RUN: {run_name}")
write_log(f"DEVICE: {device}")
write_log("")
write_log("GENERAL INFOS")
write_log(f"train length: {len(train_text)}")
write_log(f"val length: {len(val_text)}")
write_log(f"test length: {len(test_text)}")
write_log(f"vocab size: {vocab_size}")
write_log(f"context length: {CONTEXT_LENGTH}")
write_log(f"batch size: {BATCH_SIZE}")
write_log(f"n_embd: {N_EMBD}")
write_log(f"n_head: {N_HEAD}")
write_log(f"n_layer: {N_LAYER}")
write_log(f"dropout: {DROPOUT}")
write_log(f"learning rate: {LEARNING_RATE}")
write_log(f"num steps: {NUM_STEPS}")
write_log(f"eval interval: {EVAL_INTERVAL}")
write_log(f"eval iters: {EVAL_ITERS}")
write_log(f"sample every: {SAMPLE_EVERY}")
write_log(f"sample tokens: {SAMPLE_TOKENS}")
write_log(f"prompt: {repr(PROMPT)}")
write_log("")

start_time = time.time()
best_val_loss = float("inf")

for step in range(NUM_STEPS):
    if step % EVAL_INTERVAL == 0:
        losses, bpcs = estimate_loss()

        print(
            f"step {step:03d} | "
            f"train loss {losses['train']:.4f} | "
            f"val loss {losses['val']:.4f} | "
            f"train bpc {bpcs['train']:.4f} | "
            f"val bpc {bpcs['val']:.4f}"
        )

        with open(metrics_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                step,
                losses["train"],
                losses["val"],
                bpcs["train"],
                bpcs["val"],
            ])

        write_log(f"==== STEP {step:04d} ====")
        write_log(f"train loss: {losses['train']:.4f}")
        write_log(f"val loss: {losses['val']:.4f}")
        write_log(f"train bpc: {bpcs['train']:.4f}")
        write_log(f"val bpc: {bpcs['val']:.4f}")

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            write_log(f"best val loss so far: {best_val_loss:.4f}")

            if SAVE_BEST_MODEL:
                save_checkpoint(best_model_file, step, best_val_loss)
                write_log(f"best model checkpoint saved to: {best_model_file}")

        if step % SAMPLE_EVERY == 0:
            sample = generate_sample()
            write_log("")
            write_log("SAMPLE:")
            write_log(sample)
            write_log("")

        if step > 0 and step % CHECKPOINT_EVERY == 0:
            checkpoint_path = checkpoints_dir / f"checkpoint_step_{step:04d}.pt"
            save_checkpoint(checkpoint_path, step, best_val_loss)
            write_log(f"checkpoint saved to: {checkpoint_path}")

        write_log("")

    xb, yb = get_batch("train")
    xb, yb = xb.to(device), yb.to(device)

    loss = compute_loss(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

save_checkpoint(final_model_file, NUM_STEPS, best_val_loss)
end_time = time.time()
elapsed_seconds = end_time - start_time
elapsed_minutes = elapsed_seconds / 60

write_log("==== FINAL ====")
write_log(f"training time (seconds): {elapsed_seconds:.2f}")
write_log(f"training time (minutes): {elapsed_minutes:.2f}")
write_log(f"best val loss: {best_val_loss:.4f}")
write_log(f"final model checkpoint saved to: {final_model_file}")

final_sample = generate_sample()
write_log("")
write_log("FINAL SAMPLE:")
write_log(final_sample)
write_log("")

print(f"Final model saved to {final_model_file}")
if SAVE_BEST_MODEL:
    print(f"Best model saved to {best_model_file}")

print(f"Training time: {elapsed_minutes:.2f} minutes")
print(f"Run log saved to {log_file}")
print(f"Metrics saved to {metrics_file}")
import argparse
import json
from pathlib import Path

import torch

from data import encode, decode, CONTEXT_LENGTH
from model import CharGPT


device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--corpus_info_path", type=str, default="corpus_analysis/CORPUS_INFO.json")

    parser.add_argument("--prompt", type=str, default="La notte ")
    parser.add_argument("--num_poems", type=int, default=3)
    parser.add_argument("--base_seed", type=int, default=1234)

    parser.add_argument("--baseline_temperature", type=float, default=1.0)
    parser.add_argument("--baseline_top_k", type=int, default=0)
    parser.add_argument("--baseline_top_p", type=float, default=1.0)

    parser.add_argument("--temperature_values", type=str, default="0.8,1.0,1.2")
    parser.add_argument("--top_k_values", type=str, default="10,20,50")
    parser.add_argument("--top_p_values", type=str, default="0.8,0.9,0.95")

    parser.add_argument("--stanzas_to_generate", type=float, default=2.0)
    parser.add_argument("--max_new_tokens", type=int, default=0)

    return parser.parse_args()


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip() != ""]


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip() != ""]


def load_corpus_info(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"CORPUS_INFO file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_default_max_new_tokens(corpus_info, stanzas_to_generate):
    mean_line_length = corpus_info["line_statistics"]["mean_line_length"]
    mean_lines_per_stanza = corpus_info["stanza_statistics"]["mean_lines_per_stanza"]

    chars_per_line_with_newline = mean_line_length + 1
    estimated_chars = stanzas_to_generate * mean_lines_per_stanza * chars_per_line_with_newline

    return int(round(estimated_chars))


def sample_next_token(logits, temperature=1.0, top_k=0, top_p=1.0):
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    logits = logits / temperature

    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        threshold = values[:, [-1]]
        logits = torch.where(
            logits < threshold,
            torch.full_like(logits, float("-inf")),
            logits,
        )

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_mask = cumulative_probs > top_p
        sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
        sorted_mask[:, 0] = False

        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

        logits_filtered = torch.full_like(logits, float("-inf"))
        logits_filtered.scatter_(1, sorted_indices, sorted_logits)
        logits = logits_filtered

    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


@torch.no_grad()
def generate_one_poem(model, prompt, max_new_tokens, temperature, top_k, top_p, seed):
    model.eval()

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if prompt == "":
        prompt = "\n"

    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        context_cond = context[:, -CONTEXT_LENGTH:]
        logits = model(context_cond)
        logits_last = logits[:, -1, :]

        next_token = sample_next_token(
            logits=logits_last,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        context = torch.cat([context, next_token], dim=1)

    text = decode(context[0].tolist())
    return text.strip()


def generate_poem_batch(model, prompt, max_new_tokens, temperature, top_k, top_p, num_poems, base_seed):
    poems = []
    for poem_idx in range(num_poems):
        seed = base_seed + poem_idx
        poem = generate_one_poem(
            model=model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )
        poems.append(poem)
    return poems


def write_section(
    f,
    section_title,
    variable_name,
    variable_value,
    checkpoint_path,
    prompt,
    max_new_tokens,
    temperature,
    top_k,
    top_p,
    poems,
):
    f.write(f"EXPERIMENT: {section_title}\n")
    f.write(f"{variable_name}: {variable_value}\n")
    f.write(f"checkpoint_path: {checkpoint_path}\n")
    f.write(f"prompt: {repr(prompt)}\n")
    f.write(f"max_new_tokens: {max_new_tokens}\n")
    f.write(f"temperature: {temperature}\n")
    f.write(f"top_k: {top_k}\n")
    f.write(f"top_p: {top_p}\n")
    f.write("\n")

    for i, poem in enumerate(poems, start=1):
        f.write(f"POEM {i}\n")
        f.write(poem)
        f.write("\n\n")
        f.write("-" * 80)
        f.write("\n\n")

    f.write("=" * 120)
    f.write("\n\n")


def main():
    args = parse_args()

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    corpus_info = load_corpus_info(Path(args.corpus_info_path))

    if args.max_new_tokens > 0:
        max_new_tokens = args.max_new_tokens
    else:
        max_new_tokens = compute_default_max_new_tokens(
            corpus_info=corpus_info,
            stanzas_to_generate=args.stanzas_to_generate,
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = CharGPT().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    run_dir = checkpoint_path.parent
    generated_dir = run_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    temperature_values = parse_float_list(args.temperature_values)
    top_k_values = parse_int_list(args.top_k_values)
    top_p_values = parse_float_list(args.top_p_values)

    baseline_file = generated_dir / "baseline_generated.txt"
    with open(baseline_file, "w", encoding="utf-8") as f:
        poems = generate_poem_batch(
            model=model,
            prompt=args.prompt,
            max_new_tokens=max_new_tokens,
            temperature=args.baseline_temperature,
            top_k=args.baseline_top_k,
            top_p=args.baseline_top_p,
            num_poems=args.num_poems,
            base_seed=args.base_seed,
        )
        write_section(
            f=f,
            section_title="BASELINE",
            variable_name="baseline",
            variable_value="default",
            checkpoint_path=str(checkpoint_path),
            prompt=args.prompt,
            max_new_tokens=max_new_tokens,
            temperature=args.baseline_temperature,
            top_k=args.baseline_top_k,
            top_p=args.baseline_top_p,
            poems=poems,
        )
    print(f"Saved: {baseline_file}")

    temperature_file = generated_dir / "temperature_variation_generated.txt"
    with open(temperature_file, "w", encoding="utf-8") as f:
        for value in temperature_values:
            poems = generate_poem_batch(
                model=model,
                prompt=args.prompt,
                max_new_tokens=max_new_tokens,
                temperature=value,
                top_k=args.baseline_top_k,
                top_p=args.baseline_top_p,
                num_poems=args.num_poems,
                base_seed=args.base_seed,
            )
            write_section(
                f=f,
                section_title="TEMPERATURE_VARIATION",
                variable_name="temperature",
                variable_value=value,
                checkpoint_path=str(checkpoint_path),
                prompt=args.prompt,
                max_new_tokens=max_new_tokens,
                temperature=value,
                top_k=args.baseline_top_k,
                top_p=args.baseline_top_p,
                poems=poems,
            )
    print(f"Saved: {temperature_file}")

    top_k_file = generated_dir / "top_k_variation_generated.txt"
    with open(top_k_file, "w", encoding="utf-8") as f:
        for value in top_k_values:
            poems = generate_poem_batch(
                model=model,
                prompt=args.prompt,
                max_new_tokens=max_new_tokens,
                temperature=args.baseline_temperature,
                top_k=value,
                top_p=args.baseline_top_p,
                num_poems=args.num_poems,
                base_seed=args.base_seed,
            )
            write_section(
                f=f,
                section_title="TOP_K_VARIATION",
                variable_name="top_k",
                variable_value=value,
                checkpoint_path=str(checkpoint_path),
                prompt=args.prompt,
                max_new_tokens=max_new_tokens,
                temperature=args.baseline_temperature,
                top_k=value,
                top_p=args.baseline_top_p,
                poems=poems,
            )
    print(f"Saved: {top_k_file}")

    top_p_file = generated_dir / "top_p_variation_generated.txt"
    with open(top_p_file, "w", encoding="utf-8") as f:
        for value in top_p_values:
            poems = generate_poem_batch(
                model=model,
                prompt=args.prompt,
                max_new_tokens=max_new_tokens,
                temperature=args.baseline_temperature,
                top_k=args.baseline_top_k,
                top_p=value,
                num_poems=args.num_poems,
                base_seed=args.base_seed,
            )
            write_section(
                f=f,
                section_title="TOP_P_VARIATION",
                variable_name="top_p",
                variable_value=value,
                checkpoint_path=str(checkpoint_path),
                prompt=args.prompt,
                max_new_tokens=max_new_tokens,
                temperature=args.baseline_temperature,
                top_k=args.baseline_top_k,
                top_p=value,
                poems=poems,
            )
    print(f"Saved: {top_p_file}")


if __name__ == "__main__":
    main()
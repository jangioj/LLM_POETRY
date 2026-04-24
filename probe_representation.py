import argparse
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import train_text, val_text, test_text, encode, CONTEXT_LENGTH
from model import CharGPT


device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--min_line_length", type=int, default=8)
    parser.add_argument("--top_endings_k", type=int, default=8)
    parser.add_argument("--random_seed", type=int, default=42)

    return parser.parse_args()


def get_text_for_split(split: str) -> str:
    if split == "train":
        return train_text
    if split == "val":
        return val_text
    if split == "test":
        return test_text
    return train_text + "\n" + val_text + "\n" + test_text


def prepare_lines(text: str, min_line_length: int):
    raw_lines = text.splitlines()
    lines = []

    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if len(stripped) < min_line_length:
            continue
        if len(stripped) > CONTEXT_LENGTH:
            continue
        lines.append(stripped)

    return lines


def build_labels(lines, top_endings_k: int):
    punctuation_labels = []
    line_lengths = []
    endings = []

    for line in lines:
        punctuation_labels.append("punct" if line[-1] in ",.;:!?…" else "no_punct")
        line_lengths.append(len(line))
        endings.append(line[-3:] if len(line) >= 3 else line)

    q1 = np.percentile(line_lengths, 33)
    q2 = np.percentile(line_lengths, 66)

    length_bin_labels = []
    for length in line_lengths:
        if length <= q1:
            length_bin_labels.append("short")
        elif length <= q2:
            length_bin_labels.append("medium")
        else:
            length_bin_labels.append("long")

    ending_counter = Counter(endings)
    top_endings = {ending for ending, _ in ending_counter.most_common(top_endings_k)}

    ending_labels = []
    for ending in endings:
        if ending in top_endings:
            ending_labels.append(ending)
        else:
            ending_labels.append("OTHER")

    return {
        "punctuation_end": punctuation_labels,
        "length_bin": length_bin_labels,
        "ending_3chars": ending_labels,
        "line_lengths": line_lengths,
        "top_endings": sorted(top_endings),
    }


@torch.no_grad()
def extract_line_embedding(model, line: str):
    idx = torch.tensor([encode(line)], dtype=torch.long, device=device)

    B, T = idx.shape
    tok_emb = model.token_embedding_table(idx)
    pos = torch.arange(T, device=idx.device)
    pos_emb = model.position_embedding_table(pos)

    x = tok_emb + pos_emb
    x = model.blocks(x)
    x = model.ln_f(x)

    mean_pool = x.mean(dim=1).squeeze(0)
    last_token = x[:, -1, :].squeeze(0)

    return mean_pool.cpu().numpy(), last_token.cpu().numpy()


def extract_embeddings(model, lines):
    mean_embeddings = []
    last_embeddings = []

    for line in lines:
        mean_emb, last_emb = extract_line_embedding(model, line)
        mean_embeddings.append(mean_emb)
        last_embeddings.append(last_emb)

    return np.array(mean_embeddings), np.array(last_embeddings)


def run_probe(X, y, random_seed):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_seed,
        stratify=y,
    )

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=random_seed),
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=False)

    majority = Counter(y_test).most_common(1)[0][1] / len(y_test)

    return {
        "accuracy": round(acc, 4),
        "majority_baseline": round(majority, 4),
        "classification_report": report,
    }


def save_probe_txt(path: Path, metadata: dict, probe_results: dict):
    with open(path, "w", encoding="utf-8") as f:
        f.write("PROBING_ANALYSIS\n\n")

        f.write("METADATA\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

        for rep_type, tasks in probe_results.items():
            f.write(f"\nREPRESENTATION_TYPE: {rep_type}\n")
            for task_name, result in tasks.items():
                f.write(f"\nTASK: {task_name}\n")
                f.write(f"accuracy: {result['accuracy']}\n")
                f.write(f"majority_baseline: {result['majority_baseline']}\n")
                f.write("classification_report:\n")
                f.write(result["classification_report"])
                f.write("\n")


def save_probe_json(path: Path, metadata: dict, probe_results: dict):
    serializable = {
        "metadata": metadata,
        "results": {},
    }

    for rep_type, tasks in probe_results.items():
        serializable["results"][rep_type] = {}
        for task_name, result in tasks.items():
            serializable["results"][rep_type][task_name] = {
                "accuracy": result["accuracy"],
                "majority_baseline": result["majority_baseline"],
                "classification_report": result["classification_report"],
            }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=4, ensure_ascii=False)


def save_pca_plot(path: Path, X, labels, title: str, max_points=1200):
    if len(X) > max_points:
        idx = np.random.choice(len(X), size=max_points, replace=False)
        X = X[idx]
        labels = np.array(labels)[idx]
    else:
        labels = np.array(labels)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    unique_labels = sorted(set(labels))

    plt.figure(figsize=(8, 6))
    for label in unique_labels:
        mask = labels == label
        plt.scatter(coords[mask, 0], coords[mask, 1], s=12, alpha=0.7, label=str(label))

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    args = parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    run_dir = checkpoint_path.parent
    analysis_dir = run_dir / "analysis"
    probing_dir = analysis_dir / "probing"
    probing_dir.mkdir(parents=True, exist_ok=True)

    text = get_text_for_split(args.split)
    lines = prepare_lines(text, args.min_line_length)

    if len(lines) == 0:
        raise ValueError("No valid lines found after filtering.")

    if len(lines) > args.max_samples:
        lines = random.sample(lines, args.max_samples)

    labels = build_labels(lines, args.top_endings_k)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = CharGPT().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean_embeddings, last_embeddings = extract_embeddings(model, lines)

    probe_results = {
        "mean_pool": {
            "punctuation_end": run_probe(mean_embeddings, labels["punctuation_end"], args.random_seed),
            "length_bin": run_probe(mean_embeddings, labels["length_bin"], args.random_seed),
            "ending_3chars": run_probe(mean_embeddings, labels["ending_3chars"], args.random_seed),
        },
        "last_token": {
            "punctuation_end": run_probe(last_embeddings, labels["punctuation_end"], args.random_seed),
            "length_bin": run_probe(last_embeddings, labels["length_bin"], args.random_seed),
            "ending_3chars": run_probe(last_embeddings, labels["ending_3chars"], args.random_seed),
        },
    }

    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "split": args.split,
        "context_length": CONTEXT_LENGTH,
        "num_lines_used": len(lines),
        "min_line_length": args.min_line_length,
        "top_endings_k": args.top_endings_k,
        "random_seed": args.random_seed,
    }

    txt_path = probing_dir / "probing_report.txt"
    json_path = probing_dir / "probing_report.json"

    save_probe_txt(txt_path, metadata, probe_results)
    save_probe_json(json_path, metadata, probe_results)

    save_pca_plot(
        probing_dir / "pca_length_bin_mean_pool.png",
        mean_embeddings,
        labels["length_bin"],
        "PCA of mean-pooled embeddings by line-length bin",
    )

    save_pca_plot(
        probing_dir / "pca_punctuation_mean_pool.png",
        mean_embeddings,
        labels["punctuation_end"],
        "PCA of mean-pooled embeddings by punctuation ending",
    )

    print(f"Saved: {txt_path}")
    print(f"Saved: {json_path}")
    print(f"Saved PCA plots in: {probing_dir}")


if __name__ == "__main__":
    main()
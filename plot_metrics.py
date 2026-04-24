import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    return parser.parse_args()


def load_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    steps = []
    train_loss = []
    val_loss = []
    train_bpc = []
    val_bpc = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            steps.append(int(row["step"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            train_bpc.append(float(row["train_bpc"]))
            val_bpc.append(float(row["val_bpc"]))

    return {
        "steps": steps,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_bpc": train_bpc,
        "val_bpc": val_bpc,
    }


def plot_loss(metrics: dict, save_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(metrics["steps"], metrics["train_loss"], label="Train loss")
    plt.plot(metrics["steps"], metrics["val_loss"], label="Validation loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_bpc(metrics: dict, save_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(metrics["steps"], metrics["train_bpc"], label="Train BPC")
    plt.plot(metrics["steps"], metrics["val_bpc"], label="Validation BPC")
    plt.xlabel("Step")
    plt.ylabel("Bits per character")
    plt.title("Training and validation BPC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    args = parse_args()

    run_dir = Path(args.run_dir)
    metrics_file = run_dir / "metrics.csv"

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    loss_plot_file = analysis_dir / "loss_plot.png"
    bpc_plot_file = analysis_dir / "bpc_plot.png"

    metrics = load_metrics(metrics_file)
    plot_loss(metrics, loss_plot_file)
    plot_bpc(metrics, bpc_plot_file)

    print(f"Loss plot saved to: {loss_plot_file}")
    print(f"BPC plot saved to: {bpc_plot_file}")


if __name__ == "__main__":
    main()
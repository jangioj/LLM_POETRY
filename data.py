from pathlib import Path
import torch

DATA_DIR = Path("corpus")
TRAIN_FILE = DATA_DIR / "train.txt"
VAL_FILE = DATA_DIR / "val.txt"
TEST_FILE = DATA_DIR / "test.txt"

CONTEXT_LENGTH = 256
BATCH_SIZE = 32

def load_text(path):
    """
    LOAD TEXT FILES
    IN: path to text file
    OUT: text file
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text = path.read_text(encoding="utf-8")

    if len(text) == 0:
        raise ValueError(f"File is empty: {path}")

    return text

train_text = load_text(TRAIN_FILE)
val_text = load_text(VAL_FILE)
test_text = load_text(TEST_FILE)

chars = sorted(set(train_text + val_text + test_text))

#Voc: char --> index
char2idx = {ch: i for i, ch in enumerate(chars)}
#Voc: index --> char
idx2char = {i: ch for i, ch in enumerate(chars)}

vocab_size = len(chars)

def encode(text: str) -> list[int]:
    return [char2idx[ch] for ch in text]

def decode(indices: list[int]) -> str:
    return "".join(idx2char[i] for i in indices)

train_ids = torch.tensor(encode(train_text), dtype=torch.long)
val_ids = torch.tensor(encode(val_text), dtype=torch.long)
test_ids = torch.tensor(encode(test_text), dtype=torch.long)

def get_data(split: str) -> torch.Tensor:
    if split == "train":
        return train_ids
    elif split == "val":
        return val_ids
    elif split == "test":
        return test_ids
    else:
        raise ValueError("split must be 'train', 'val', or 'test'")

def get_batch(split: str):
    data = get_data(split)

    max_start = len(data) - CONTEXT_LENGTH - 1
    starts = torch.randint(0, max_start + 1, (BATCH_SIZE,))

    x = torch.stack([data[i:i + CONTEXT_LENGTH] for i in starts])
    y = torch.stack([data[i + 1:i + CONTEXT_LENGTH + 1] for i in starts])

    return x, y
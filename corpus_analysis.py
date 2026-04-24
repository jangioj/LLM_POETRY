from pathlib import Path
from statistics import mean, median
from collections import Counter
import json

from data import train_text, val_text, test_text


OUTPUT_DIR = Path("corpus/corpus_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TXT_FILE = OUTPUT_DIR / "CORPUS_INFO.txt"
JSON_FILE = OUTPUT_DIR / "CORPUS_INFO.json"


def percentile(sorted_values, p):
    if not sorted_values:
        return None
    index = int(p * (len(sorted_values) - 1))
    return sorted_values[index]


def safe_mean(values):
    return round(mean(values), 2) if values else None


def safe_median(values):
    return median(values) if values else None


full_text = train_text + val_text + test_text

all_chars = list(full_text)
char_counter = Counter(all_chars)

lines = full_text.splitlines()
non_empty_lines = [line for line in lines if line.strip() != ""]

line_lengths = [len(line) for line in non_empty_lines]
sorted_line_lengths = sorted(line_lengths)

blank_line_count = len(lines) - len(non_empty_lines)

stanzas = [stanza for stanza in full_text.split("\n\n") if stanza.strip() != ""]
stanza_line_counts = [
    len([line for line in stanza.splitlines() if line.strip() != ""])
    for stanza in stanzas
]
sorted_stanza_line_counts = sorted(stanza_line_counts)

words = full_text.split()
word_lengths = [len(word) for word in words]
sorted_word_lengths = sorted(word_lengths)

line_endings = []
for line in non_empty_lines:
    stripped = line.strip()
    if len(stripped) >= 3:
        line_endings.append(stripped[-3:])
    else:
        line_endings.append(stripped)

line_ending_counter = Counter(line_endings)

top_characters = char_counter.most_common(30)
top_line_endings = line_ending_counter.most_common(30)

info = {
    "split_sizes_chars": {
        "train": len(train_text),
        "val": len(val_text),
        "test": len(test_text),
        "total": len(full_text),
    },
    "global_counts": {
        "total_characters": len(full_text),
        "total_lines": len(lines),
        "non_empty_lines": len(non_empty_lines),
        "blank_lines": blank_line_count,
        "total_words": len(words),
        "vocabulary_size_char_level": len(char_counter),
    },
    "line_statistics": {
        "mean_line_length": safe_mean(line_lengths),
        "median_line_length": safe_median(line_lengths),
        "min_line_length": min(line_lengths) if line_lengths else None,
        "max_line_length": max(line_lengths) if line_lengths else None,
        "p90_line_length": percentile(sorted_line_lengths, 0.90),
        "p95_line_length": percentile(sorted_line_lengths, 0.95),
        "p99_line_length": percentile(sorted_line_lengths, 0.99),
    },
    "word_statistics": {
        "mean_word_length": safe_mean(word_lengths),
        "median_word_length": safe_median(word_lengths),
        "min_word_length": min(word_lengths) if word_lengths else None,
        "max_word_length": max(word_lengths) if word_lengths else None,
        "p90_word_length": percentile(sorted_word_lengths, 0.90),
        "p95_word_length": percentile(sorted_word_lengths, 0.95),
        "p99_word_length": percentile(sorted_word_lengths, 0.99),
    },
    "stanza_statistics": {
        "estimated_stanza_count": len(stanzas),
        "mean_lines_per_stanza": safe_mean(stanza_line_counts),
        "median_lines_per_stanza": safe_median(stanza_line_counts),
        "min_lines_per_stanza": min(stanza_line_counts) if stanza_line_counts else None,
        "max_lines_per_stanza": max(stanza_line_counts) if stanza_line_counts else None,
        "p90_lines_per_stanza": percentile(sorted_stanza_line_counts, 0.90),
        "p95_lines_per_stanza": percentile(sorted_stanza_line_counts, 0.95),
    },
    "structural_ratios": {
        "blank_line_ratio": round(blank_line_count / len(lines), 4) if lines else None,
        "non_empty_line_ratio": round(len(non_empty_lines) / len(lines), 4) if lines else None,
    },
    "top_characters": top_characters,
    "top_line_endings_last_3_chars": top_line_endings,
}


with open(TXT_FILE, "w", encoding="utf-8") as f:
    f.write("CORPUS_INFO\n\n")

    f.write("SPLIT_SIZES_CHARS\n")
    for key, value in info["split_sizes_chars"].items():
        f.write(f"{key}: {value}\n")

    f.write("\nGLOBAL_COUNTS\n")
    for key, value in info["global_counts"].items():
        f.write(f"{key}: {value}\n")

    f.write("\nLINE_STATISTICS\n")
    for key, value in info["line_statistics"].items():
        f.write(f"{key}: {value}\n")

    f.write("\nWORD_STATISTICS\n")
    for key, value in info["word_statistics"].items():
        f.write(f"{key}: {value}\n")

    f.write("\nSTANZA_STATISTICS\n")
    for key, value in info["stanza_statistics"].items():
        f.write(f"{key}: {value}\n")

    f.write("\nSTRUCTURAL_RATIOS\n")
    for key, value in info["structural_ratios"].items():
        f.write(f"{key}: {value}\n")

    f.write("\nTOP_CHARACTERS\n")
    for char, count in info["top_characters"]:
        f.write(f"{repr(char)}: {count}\n")

    f.write("\nTOP_LINE_ENDINGS_LAST_3_CHARS\n")
    for ending, count in info["top_line_endings_last_3_chars"]:
        f.write(f"{repr(ending)}: {count}\n")

with open(JSON_FILE, "w", encoding="utf-8") as f:
    json.dump(info, f, indent=4, ensure_ascii=False)

print(f"Saved: {TXT_FILE}")
print(f"Saved: {JSON_FILE}")
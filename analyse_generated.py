import argparse
import json
from pathlib import Path
from statistics import mean, median
from collections import Counter
import re


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--corpus_info_path", type=str, default="corpus/corpus_analysis/CORPUS_INFO.json")
    return parser.parse_args()


def percentile(sorted_values, p):
    if not sorted_values:
        return None
    index = int(p * (len(sorted_values) - 1))
    return sorted_values[index]


def safe_mean(values):
    return round(mean(values), 4) if values else None


def safe_median(values):
    return median(values) if values else None


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_sections(text):
    pattern = r"EXPERIMENT:\s+(.*?)\n(.*?)(?=\n={40,}|\Z)"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    sections = []

    for experiment_name, content in matches:
        sections.append({
            "experiment_name": experiment_name.strip(),
            "content": content.strip(),
        })

    return sections


def extract_variable(content):
    lines = content.splitlines()
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key in {"temperature", "top_k", "top_p", "baseline"}:
                return key, value
    return "unknown", "unknown"


def extract_poems(content):
    poems = []
    pattern = r"POEM\s+\d+\n(.*?)(?=\n-{20,}|\Z)"
    matches = re.findall(pattern, content, flags=re.DOTALL)

    for match in matches:
        poem = match.strip()
        if poem:
            poems.append(poem)

    return poems


def get_line_endings(poems, ending_size=3):
    endings = []

    for poem in poems:
        lines = poem.splitlines()
        non_empty_lines = [line.strip() for line in lines if line.strip() != ""]

        for line in non_empty_lines:
            endings.append(line[-ending_size:] if len(line) >= ending_size else line)

    return endings


def analyze_poems(poems):
    poem_char_lengths = [len(poem) for poem in poems]
    poem_line_counts = [len(poem.splitlines()) for poem in poems]

    all_lines = []
    non_empty_lines = []
    blank_lines = 0

    for poem in poems:
        poem_lines = poem.splitlines()
        all_lines.extend(poem_lines)

    for line in all_lines:
        if line.strip() == "":
            blank_lines += 1
        else:
            non_empty_lines.append(line)

    line_lengths = [len(line) for line in non_empty_lines]
    word_list = " ".join(poems).split()
    word_lengths = [len(word) for word in word_list]

    line_endings = get_line_endings(poems, ending_size=3)
    line_ending_counter = Counter(line_endings)

    sorted_line_lengths = sorted(line_lengths)
    sorted_poem_char_lengths = sorted(poem_char_lengths)
    sorted_poem_line_counts = sorted(poem_line_counts)
    sorted_word_lengths = sorted(word_lengths)

    total_lines = len(all_lines)
    non_empty_line_count = len(non_empty_lines)

    return {
        "global_counts": {
            "num_poems": len(poems),
            "total_characters": sum(poem_char_lengths),
            "total_lines": total_lines,
            "non_empty_lines": non_empty_line_count,
            "blank_lines": blank_lines,
            "total_words": len(word_list),
        },
        "poem_length_statistics_chars": {
            "mean": safe_mean(poem_char_lengths),
            "median": safe_median(poem_char_lengths),
            "min": min(poem_char_lengths) if poem_char_lengths else None,
            "max": max(poem_char_lengths) if poem_char_lengths else None,
            "p90": percentile(sorted_poem_char_lengths, 0.90),
            "p95": percentile(sorted_poem_char_lengths, 0.95),
        },
        "poem_length_statistics_lines": {
            "mean": safe_mean(poem_line_counts),
            "median": safe_median(poem_line_counts),
            "min": min(poem_line_counts) if poem_line_counts else None,
            "max": max(poem_line_counts) if poem_line_counts else None,
            "p90": percentile(sorted_poem_line_counts, 0.90),
            "p95": percentile(sorted_poem_line_counts, 0.95),
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
        "structural_ratios": {
            "blank_line_ratio": round(blank_lines / total_lines, 4) if total_lines > 0 else None,
            "non_empty_line_ratio": round(non_empty_line_count / total_lines, 4) if total_lines > 0 else None,
        },
        "top_line_endings_last_3_chars": line_ending_counter.most_common(20),
    }


def build_comparison(generated_stats, corpus_info):
    corpus_line = corpus_info["line_statistics"]
    corpus_word = corpus_info["word_statistics"]
    corpus_stanza = corpus_info["stanza_statistics"]
    corpus_structural = corpus_info["structural_ratios"]

    generated_line = generated_stats["line_statistics"]
    generated_word = generated_stats["word_statistics"]
    generated_poem_lines = generated_stats["poem_length_statistics_lines"]
    generated_structural = generated_stats["structural_ratios"]
    generated_global = generated_stats["global_counts"]

    return {
        "generated_vs_corpus": {
            "mean_line_length_difference": round(
                generated_line["mean_line_length"] - corpus_line["mean_line_length"], 4
            ) if generated_line["mean_line_length"] is not None else None,
            "median_line_length_difference": (
                generated_line["median_line_length"] - corpus_line["median_line_length"]
                if generated_line["median_line_length"] is not None else None
            ),
            "mean_word_length_difference": round(
                generated_word["mean_word_length"] - corpus_word["mean_word_length"], 4
            ) if generated_word["mean_word_length"] is not None else None,
            "blank_line_ratio_difference": round(
                generated_structural["blank_line_ratio"] - corpus_structural["blank_line_ratio"], 4
            ) if generated_structural["blank_line_ratio"] is not None else None,
            "mean_lines_per_poem_vs_mean_lines_per_stanza_difference": round(
                generated_poem_lines["mean"] - corpus_stanza["mean_lines_per_stanza"], 4
            ) if generated_poem_lines["mean"] is not None else None,
        },
        "corpus_reference": {
            "mean_line_length": corpus_line["mean_line_length"],
            "median_line_length": corpus_line["median_line_length"],
            "mean_word_length": corpus_word["mean_word_length"],
            "blank_line_ratio": corpus_structural["blank_line_ratio"],
            "mean_lines_per_stanza": corpus_stanza["mean_lines_per_stanza"],
        },
        "generated_reference": {
            "num_poems": generated_global["num_poems"],
            "mean_line_length": generated_line["mean_line_length"],
            "median_line_length": generated_line["median_line_length"],
            "mean_word_length": generated_word["mean_word_length"],
            "blank_line_ratio": generated_structural["blank_line_ratio"],
            "mean_lines_per_poem": generated_poem_lines["mean"],
        },
    }


def write_txt_report(path: Path, source_file: str, section_results: list):
    with open(path, "w", encoding="utf-8") as f:
        f.write("GENERATED_ANALYSIS\n\n")
        f.write(f"source_file: {source_file}\n\n")

        for item in section_results:
            f.write(f"SECTION: {item['experiment_name']}\n")
            f.write(f"variable_name: {item['variable_name']}\n")
            f.write(f"variable_value: {item['variable_value']}\n\n")

            f.write("GLOBAL_COUNTS\n")
            for key, value in item["generated_stats"]["global_counts"].items():
                f.write(f"{key}: {value}\n")

            f.write("\nPOEM_LENGTH_STATISTICS_CHARS\n")
            for key, value in item["generated_stats"]["poem_length_statistics_chars"].items():
                f.write(f"{key}: {value}\n")

            f.write("\nPOEM_LENGTH_STATISTICS_LINES\n")
            for key, value in item["generated_stats"]["poem_length_statistics_lines"].items():
                f.write(f"{key}: {value}\n")

            f.write("\nLINE_STATISTICS\n")
            for key, value in item["generated_stats"]["line_statistics"].items():
                f.write(f"{key}: {value}\n")

            f.write("\nWORD_STATISTICS\n")
            for key, value in item["generated_stats"]["word_statistics"].items():
                f.write(f"{key}: {value}\n")

            f.write("\nSTRUCTURAL_RATIOS\n")
            for key, value in item["generated_stats"]["structural_ratios"].items():
                f.write(f"{key}: {value}\n")

            f.write("\nTOP_LINE_ENDINGS_LAST_3_CHARS\n")
            for ending, count in item["generated_stats"]["top_line_endings_last_3_chars"]:
                f.write(f"{repr(ending)}: {count}\n")

            f.write("\nCOMPARISON_WITH_CORPUS\n")
            for section_name, section_values in item["comparison"].items():
                f.write(f"\n{section_name.upper()}\n")
                for key, value in section_values.items():
                    f.write(f"{key}: {value}\n")

            f.write("\n")
            f.write("=" * 120)
            f.write("\n\n")


def main():
    args = parse_args()

    run_dir = Path(args.run_dir)
    generated_dir = run_dir / "generated"
    analysis_dir = run_dir / "analysis"

    if not generated_dir.exists():
        raise FileNotFoundError(f"Generated directory not found: {generated_dir}")

    analysis_dir.mkdir(parents=True, exist_ok=True)
    corpus_info = load_json(Path(args.corpus_info_path))

    txt_files = sorted(generated_dir.glob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in {generated_dir}")

    for txt_file in txt_files:
        text = txt_file.read_text(encoding="utf-8")
        sections = parse_sections(text)

        if not sections:
            print(f"Skipping {txt_file.name}: no sections found")
            continue

        section_results = []

        for section in sections:
            variable_name, variable_value = extract_variable(section["content"])
            poems = extract_poems(section["content"])

            if not poems:
                continue

            generated_stats = analyze_poems(poems)
            comparison = build_comparison(generated_stats, corpus_info)

            section_results.append({
                "experiment_name": section["experiment_name"],
                "variable_name": variable_name,
                "variable_value": variable_value,
                "generated_stats": generated_stats,
                "comparison": comparison,
            })

        if not section_results:
            print(f"Skipping {txt_file.name}: no poem sections parsed")
            continue

        stem = txt_file.stem.replace("_generated", "")
        txt_path = analysis_dir / f"{stem}_analysis.txt"

        write_txt_report(
            path=txt_path,
            source_file=txt_file.name,
            section_results=section_results,
        )

        print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
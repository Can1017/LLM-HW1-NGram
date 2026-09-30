"""全量训练集上的 1—10 阶 KenLM 实验；逐阶保存，支持中断后续跑。"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "results" / "completed"
RESULT = ROOT / "results" / "full"
MODEL = ROOT / "models" / "full"
LOG = ROOT / "logs" / "full"
WORK = Path.home() / "hw1_kenlm_full"
for path in (RESULT, MODEL, LOG, WORK):
    path.mkdir(parents=True, exist_ok=True)


def load_manifest():
    path = RESULT / "kenlm.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"corpus": "complete processed train split", "rows": []}


def save_manifest(data):
    (RESULT / "kenlm.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_input():
    source = PROC / "train.txt"
    target = WORK / "train.txt"
    if not target.exists() or target.stat().st_size != source.stat().st_size:
        shutil.copy2(source, target)
    return target


def parse_unigrams(path):
    values = {}
    in_section = False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip() == "\\1-grams:":
                in_section = True
                continue
            if in_section and line.startswith("\\"):
                break
            if in_section:
                parts = line.split()
                if len(parts) >= 2:
                    values[parts[1]] = float(parts[0])
    return values


def evaluate(path, order):
    if order == 1:
        unigram = parse_unigrams(path)
        default = unigram["<unk>"]
        def sentence_score(text):
            return math.fsum(unigram.get(token, default) for token in text.split() + ["</s>"])
    else:
        import kenlm
        model = kenlm.Model(str(path))
        def sentence_score(text):
            return model.score(text, bos=True, eos=True)
    metrics = {}
    for split in ("valid", "test"):
        total_log10 = 0.0
        tokens = 0
        sentences = 0
        with (PROC / f"{split}.txt").open("r", encoding="utf-8") as handle:
            for text in handle:
                text = text.strip()
                if not text:
                    continue
                total_log10 += sentence_score(text)
                tokens += len(text.split()) + 1
                sentences += 1
        nll = -total_log10 * math.log(10) / tokens
        metrics[split] = {"sentences": sentences, "predicted_tokens": tokens, "nll_nats_per_token": nll, "ppl": math.exp(nll)}
    if order > 1:
        del model
        gc.collect()
    return metrics


def run_order(order, train_input):
    ext_arpa = WORK / f"kenlm_{order}.arpa"
    target = MODEL / ext_arpa.name
    command = [str(ROOT / ".tools" / "kenlm" / "bin" / "lmplz"), "-o", str(order), "-S", "2G", "-T", str(WORK), "--discount_fallback", "--text", str(train_input), "--arpa", str(ext_arpa), "--verbose_header"]
    start = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    phases, lines = [], []
    for line in process.stderr:
        lines.append(line)
        match = re.search(r"===\s*([1-5])/5\s*(.*?)\s*===", line)
        if match:
            phases.append({"stage": int(match.group(1)), "name": match.group(2), "start_seconds": time.perf_counter() - start})
    exit_code = process.wait()
    elapsed = time.perf_counter() - start
    log_file = LOG / f"kenlm_{order}.stderr.log"
    log_file.write_text("".join(lines), encoding="utf-8")
    if exit_code:
        raise RuntimeError(f"{order}-gram 训练退出码 {exit_code}；请查看 {log_file}")
    phases.append({"stage": 6, "name": "完成", "start_seconds": elapsed})
    (LOG / f"kenlm_{order}.phases.json").write_text(json.dumps(phases, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(ext_arpa, target)
    with target.open("r", encoding="utf-8", errors="ignore") as stream:
        header = stream.read(5000)
    counts = {int(n): int(count) for n, count in re.findall(r"ngram (\d+)=(\d+)", header)}
    peak = re.search(r"RSSMax:(\d+) kB", "".join(lines))
    row = {"method": "KenLM MKN", "order": order, "train_seconds": elapsed, "arpa_bytes": target.stat().st_size, "peak_kb": int(peak.group(1)) if peak else None, "ngram_counts": counts, "model": str(target.relative_to(ROOT)), "log": str(log_file.relative_to(ROOT)), "command": command, "evaluation": evaluate(target, order)}
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", type=int, nargs="*", default=list(range(1, 11)))
    args = parser.parse_args()
    train_input = ensure_input()
    data = load_manifest()
    completed = {x["order"] for x in data["rows"]}
    for order in args.orders:
        if order in completed:
            print(f"{order}-gram 已完成，跳过", flush=True)
            continue
        row = run_order(order, train_input)
        data["rows"].append(row)
        data["rows"].sort(key=lambda x: x["order"])
        save_manifest(data)
        print(f"KenLM {order}-gram: {row['train_seconds']:.1f}s, test PPL={row['evaluation']['test']['ppl']:.1f}", flush=True)
    print(f"产物：{RESULT / 'kenlm.json'}", flush=True)


if __name__ == "__main__":
    main()

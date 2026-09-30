"""在相同语料上比较 KenLM 与 Stupid Backoff 的训练、预测和续写。"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
RESULT = BASE / "results" / "methods"
LOG = BASE / "logs" / "methods"
MODEL = BASE / "models" / "methods"
for folder in (RESULT, LOG, MODEL):
    folder.mkdir(parents=True, exist_ok=True)
PROMPT_TEXT = "在阳光明媚的五月，我们学校胜利召开了"
PROMPT = "在 阳光明媚 的 五月 ， 我们 学校 胜利 召开 了".split()
ORDERS = (2, 3, 5)
BIN = BASE / ".tools" / "kenlm" / "bin"


def emit(name, value):
    (RESULT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def lines_for_experiment():
    lines = (BASE / "results" / "completed" / "train.txt").read_text(encoding="utf-8").splitlines()
    selected = sorted(random.Random(2026).sample(range(len(lines)), min(30000, len(lines))))
    sample = [lines[i] for i in selected]
    path = RESULT / "train_sample.txt"
    path.write_text("\n".join(sample) + "\n", encoding="utf-8")
    check = (BASE / "results" / "completed" / "valid.txt").read_text(encoding="utf-8").splitlines()
    return sample, check


def train_kenlm(sample_path, orders):
    # KenLM 的 mmap 临时文件在 WSL 的 DrvFs 路径会触发段错误；先复制到本机 WSL ext4。
    work = Path.home() / "hw1_kenlm_work"
    work.mkdir(exist_ok=True)
    ext4_sample = work / "train_sample.txt"
    shutil.copy2(sample_path, ext4_sample)
    out = []
    for order in orders:
        times = []
        arpa = MODEL / f"kenlm_{order}.arpa"
        for repeat in range(3):
            ext4_arpa = work / f"kenlm_{order}.arpa"
            command = [str(BIN / "lmplz"), "-o", str(order), "-S", "2G", "-T", str(work), "--text", str(ext4_sample), "--arpa", str(ext4_arpa), "--verbose_header"]
            start = time.perf_counter()
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
            phases = []
            stderr = []
            for line in process.stderr:
                stderr.append(line)
                phase = re.search(r"===\s*([1-5])/5\s*(.*?)\s*===", line)
                if phase:
                    phases.append({"stage": int(phase.group(1)), "name": phase.group(2), "start_seconds": time.perf_counter() - start})
            returncode = process.wait()
            elapsed = time.perf_counter() - start
            (LOG / f"kenlm_{order}_{repeat + 1}.stderr.log").write_text("".join(stderr), encoding="utf-8")
            if returncode:
                raise RuntimeError(f"KenLM {order} 阶训练失败，查看日志；退出码 {returncode}")
            shutil.copy2(ext4_arpa, arpa)
            phases.append({"stage": 6, "name": "完成", "start_seconds": elapsed})
            (LOG / f"kenlm_{order}_{repeat + 1}.phases.json").write_text(json.dumps(phases, ensure_ascii=False, indent=2), encoding="utf-8")
            times.append(elapsed)
            print(f"KenLM {order}-gram #{repeat+1}: {elapsed:.2f}s", flush=True)
        with arpa.open("r", encoding="utf-8") as stream:
            counts = {}
            for line in stream:
                if line.strip() == "\\1-grams:":
                    break
                if "ngram" in line and "=" in line:
                    left, right = line.strip().split("=", 1)
                    counts[left] = int(right)
        out.append({"method": "KenLM", "order": order, "seconds": times, "median_seconds": statistics.median(times), "arpa_bytes": arpa.stat().st_size, "ngram_counts": counts, "model_path": str(arpa.relative_to(BASE))})
    return out


class Stupid:
    """实现 Brants 等提出的固定系数回退；分数未归一化。"""

    def __init__(self, order, lines, alpha=.4):
        self.order = order
        self.alpha = alpha
        self.counts = [Counter() for _ in range(order + 1)]
        self.contexts = [Counter() for _ in range(order)]
        self.followers = [defaultdict(set) for _ in range(order)]
        for line in lines:
            words = line.split()
            sequence = ["<s>"] * (order - 1) + words + ["</s>"]
            for j in range(order - 1, len(sequence)):
                for n in range(1, order + 1):
                    self.counts[n][tuple(sequence[j-n+1:j+1])] += 1
        for n in range(2, order + 1):
            for gram, count in self.counts[n].items():
                self.contexts[n-1][gram[:-1]] += count
                self.followers[n-1][gram[:-1]].add(gram[-1])
        self.unigram_total = sum(self.counts[1].values())

    def score(self, word, history):
        context = tuple(history[-(self.order - 1):])
        return self._score(word, context)

    def _score(self, word, context):
        if not context:
            return self.counts[1].get((word,), 0) / self.unigram_total
        joint = self.counts[len(context) + 1].get(context + (word,), 0)
        if joint:
            return joint / self.contexts[len(context)][context]
        return self.alpha * self._score(word, context[1:])


def train_stupid(sample, orders):
    out, models = [], {}
    for order in orders:
        times = []
        for repeat in range(3):
            start = time.perf_counter()
            model = Stupid(order, sample)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"Stupid Backoff {order}-gram #{repeat+1}: {elapsed:.2f}s", flush=True)
        models[order] = model
        out.append({"method": "Stupid Backoff", "order": order, "seconds": times, "median_seconds": statistics.median(times), "ngram_counts": {f"ngram {n}": len(model.counts[n]) for n in range(1, order+1)}})
    return out, models


def make_scoring_fn(method, order, models):
    if method == "KenLM":
        import kenlm
        model = models[order]

        @functools.lru_cache(maxsize=1024)
        def state_for(history):
            state = kenlm.State()
            model.BeginSentenceWrite(state)
            for previous in history:
                newer = kenlm.State()
                model.BaseScore(state, previous, newer)
                state = newer
            return state

        def scorer(word, history):
            state = state_for(tuple(history))
            newer = kenlm.State()
            return 10 ** model.BaseScore(state, word, newer)

        return scorer
    return models[order].score


def evaluate(ken_models, stupid_models, valid, words):
    import kenlm
    frequent = [w for w, _ in words.most_common(300) if w not in {"<UNK>", "<s>"}]
    rng = random.Random(927)
    choices = []
    for line in valid[:600]:
        sequence = line.split()
        if len(sequence) < 4:
            continue
        index = rng.randrange(2, len(sequence))
        choices.append((sequence[max(0, index-4):index], sequence[index]))
    choices = choices[:150]
    rows = []
    for method, models in (("KenLM", ken_models), ("Stupid Backoff", stupid_models)):
        for order in ORDERS:
            scorer = make_scoring_fn(method, order, models)
            ranks = []
            for history, truth in choices:
                candidates = sorted(set(frequent) | {truth})
                ordered = sorted(candidates, key=lambda word: (-scorer(word, history), word))
                ranks.append(ordered.index(truth) + 1)
            row = {"method": method, "order": order, "n": len(ranks), "top1": sum(r == 1 for r in ranks)/len(ranks), "top5": sum(r <= 5 for r in ranks)/len(ranks), "mrr": statistics.mean(1/r for r in ranks), "candidate_policy": "300 frequent training unigrams plus true word"}
            if method == "KenLM":
                log10 = sum(models[order].score(line, bos=True, eos=True) for line in valid[:600])
                tokens = sum(len(line.split()) + 1 for line in valid[:600])
                row["nll_nats_per_token"] = -log10 * math.log(10) / tokens
                row["ppl"] = 10 ** (-log10 / tokens)
                row["perplexity_tokens"] = tokens
            rows.append(row)
            print(f"评测 {method} {order}: top1={rows[-1]['top1']:.3f}", flush=True)
    return rows


def generate(ken_models, stupid_models, words):
    frequent = {w for w, _ in words.most_common(300)} - {"<UNK>", "<s>"}
    rows = []
    for method, models in (("KenLM", ken_models), ("Stupid Backoff", stupid_models)):
        for order, temp, greedy in [(2, 1, False), (3, 1, False), (5, 1, False), (3, .7, False), (3, 1.3, False), (3, 1, True)]:
            scorer = make_scoring_fn(method, order, models)
            successors = stupid_models[order].followers[order - 1]
            for seed in ((42,) if greedy else (42, 43, 44)):
                rng = random.Random(seed)
                history = [w if w in words else "<UNK>" for w in PROMPT]
                output = []
                eos = False
                for step in range(80):
                    context = tuple(history[-(order-1):])
                    candidates = (frequent | successors.get(context, set()) | {"</s>"}) - {"<UNK>", "<s>"}
                    ranked = sorted(((scorer(w, history), w) for w in candidates), key=lambda item: (-item[0], item[1]))[:20]
                    if greedy:
                        chosen = ranked[0][1]
                    else:
                        weights = [max(score, 1e-300) ** (1/temp) for score, _ in ranked]
                        chosen = rng.choices([word for _, word in ranked], weights=weights)[0]
                    if chosen == "</s>":
                        eos = True
                        break
                    history.append(chosen)
                    output.append(chosen)
                rows.append({"method": method, "order": order, "temperature": temp, "greedy": greedy, "seed": seed, "length": len(output), "eos": eos, "tokens": output, "text": PROMPT_TEXT + "".join(output)})
                print(f"续写 {method} {order} T={temp} seed={seed}: {len(output)} 词", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-only", action="store_true", help="只训练并保存 KenLM 结果")
    args = parser.parse_args()
    kenlm_commit = subprocess.check_output(["git", "-C", str(BASE / "third_party" / "kenlm"), "rev-parse", "HEAD"], text=True).strip()
    memory_line = next(line.strip() for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemTotal:")) if Path("/proc/meminfo").exists() else "unknown"
    cpu_model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "unknown") if Path("/proc/cpuinfo").exists() else "unknown"
    emit("environment.json", {"platform": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count(), "cpu_model": cpu_model, "memory": memory_line, "kenlm_commit": kenlm_commit, "lmplz": str(BIN / "lmplz"), "training_staging": str(Path.home() / "hw1_kenlm_work"), "gpu_role": "unused: KenLM and Stupid Backoff CPU n-gram algorithms"})
    sample, valid = lines_for_experiment()
    counts = Counter(word for line in sample for word in line.split())
    training = train_kenlm(RESULT / "train_sample.txt", ORDERS)
    if args.train_only:
        emit("kenlm_training.json", training)
        return
    stupid_training, stupid_models = train_stupid(sample, ORDERS)
    emit("training.json", {"sample_sentences": len(sample), "sample_tokens": sum(len(x.split()) for x in sample), "trainings": training + stupid_training})
    import kenlm
    ken_models = {order: kenlm.Model(str(MODEL / f"kenlm_{order}.arpa")) for order in ORDERS}
    emit("evaluation.json", evaluate(ken_models, stupid_models, valid, counts))
    emit("generation.json", generate(ken_models, stupid_models, counts))
    print("产物：models/methods/、results/methods/、logs/methods/", flush=True)


if __name__ == "__main__":
    main()

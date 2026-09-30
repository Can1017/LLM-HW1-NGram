"""对完整语料训练的 1—10 阶模型进行同候选集排序与单种子续写。"""
from __future__ import annotations

import argparse
import functools
import gc
import json
import math
import random
import sqlite3
import struct
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FULL = ROOT / "results" / "full"
TEST = ROOT / "results" / "completed" / "test.txt"
DB = Path.home() / "hw1_stupid_full" / "stupid_counts.sqlite3"
PROMPT = "在 阳光明媚 的 五月 ， 我们 学校 胜利 召开 了".split()
PROMPT_TEXT = "在阳光明媚的五月，我们学校胜利召开了"


class Scorer:
    def __init__(self, method, order, db, word_ids):
        self.method, self.order, self.db, self.ids = method, order, db, word_ids
        if method == "KenLM":
            path = ROOT / "models" / "full" / f"kenlm_{order}.arpa"
            if order == 1:
                from run_full_kenlm import parse_unigrams
                self.unigrams = parse_unigrams(path)
            else:
                import kenlm
                self.kenlm = kenlm
                self.model = kenlm.Model(str(path))

    @functools.lru_cache(maxsize=300000)
    def count(self, gram):
        packed = struct.pack("<" + "I" * len(gram), *gram)
        row = self.db.execute(f"SELECT c FROM counts_{len(gram)} WHERE key=?", (packed,)).fetchone()
        return row[0] if row else 0

    def stupid(self, word, context):
        if not context:
            return self.count((word,)) / self.total
        joint = self.count(context + (word,))
        if joint:
            denominator = self.sentences if all(x == 0 for x in context) else self.count(context)
            return joint / denominator
        return 0.4 * self.stupid(word, context[1:])

    @functools.lru_cache(maxsize=10000)
    def state(self, history):
        state = self.kenlm.State()
        self.model.BeginSentenceWrite(state)
        for token in history:
            next_state = self.kenlm.State()
            self.model.BaseScore(state, token, next_state)
            state = next_state
        return state

    def score(self, word, history):
        if self.method == "Stupid Backoff":
            seq = tuple(self.ids.get(x, self.ids["<UNK>"]) for x in history[-(self.order-1):]) if self.order > 1 else ()
            return math.log(max(self.stupid(self.ids[word], (0,) * max(0, self.order-1-len(seq)) + seq), 1e-300))
        if self.order == 1:
            return self.unigrams.get(word, self.unigrams["<unk>"]) * math.log(10)
        state = self.state(tuple(history[-(self.order-1):]))
        next_state = self.kenlm.State()
        return self.model.BaseScore(state, word, next_state) * math.log(10)


def choose_positions():
    rng = random.Random(42)
    lines = TEST.read_text(encoding="utf-8").splitlines()
    eligible = [i for i, line in enumerate(lines) if len(line.split()) >= 3]
    indexes = rng.sample(eligible, 150)
    positions = []
    for number in indexes:
        words = lines[number].split()
        pos = rng.randrange(1, len(words))
        positions.append((words[:pos], words[pos]))
    return positions


def evaluate(scorer, positions, candidates):
    ranks = []
    for history, target in positions:
        options = sorted(set(candidates) | {target})
        scored = sorted(options, key=lambda w: (-scorer.score(w, history), w))
        ranks.append(scored.index(target) + 1)
    return {"positions": len(ranks), "top1": sum(x == 1 for x in ranks) / len(ranks),
            "top5": sum(x <= 5 for x in ranks) / len(ranks),
            "mrr": sum(1 / x for x in ranks) / len(ranks),
            "candidate_policy": "训练集高频 100 词及真实目标词；测试集随机抽取 150 个位置，随机种子 42"}


def generate(scorer, candidates, temp=1.0, greedy=False):
    rng = random.Random(42)
    history = [x if x in scorer.ids else "<UNK>" for x in PROMPT]
    output, ended = [], False
    for _ in range(80):
        scored = sorted(((scorer.score(w, history), w) for w in candidates), key=lambda x: (-x[0], x[1]))[:20]
        if greedy:
            chosen = scored[0][1]
        else:
            maximum = scored[0][0]
            weights = [math.exp((v - maximum) / temp) for v, _ in scored]
            chosen = rng.choices([w for _, w in scored], weights=weights, k=1)[0]
        if chosen == "</s>":
            ended = True
            break
        output.append(chosen)
        history.append(chosen)
    return {"temperature": temp, "greedy": greedy, "seed": 42, "length": len(output),
            "eos": ended, "tokens": output, "text": PROMPT_TEXT + "".join(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", nargs="*", type=int, default=list(range(1, 11)))
    args = parser.parse_args()
    vocab = json.loads((FULL / "stupid_vocab.json").read_text(encoding="utf-8"))
    ids = {w: i for i, w in enumerate(vocab["words"])}
    counter = Counter()
    with (ROOT / "results" / "completed" / "train.txt").open(encoding="utf-8") as file:
        for line in file:
            counter.update(line.split())
    frequent = [w for w, _ in counter.most_common(200) if w not in {"<UNK>", "<s>", "</s>"}]
    eval_candidates = frequent[:100]
    generation_candidates = sorted(set(frequent) | {"</s>"})
    positions = choose_positions()
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows, outputs = [], []
    for order in args.orders:
        for method in ("KenLM", "Stupid Backoff"):
            scorer = Scorer(method, order, db, ids)
            scorer.total = vocab["tokens"] + vocab["sentences"]
            scorer.sentences = vocab["sentences"]
            result = evaluate(scorer, positions, eval_candidates)
            result.update({"method": method, "order": order})
            rows.append(result)
            variants = [(1.0, False)] + ([(0.7, False), (1.3, False), (1.0, True)] if order == 5 else [])
            for temp, greedy in variants:
                record = generate(scorer, generation_candidates, temp, greedy)
                record.update({"method": method, "order": order})
                outputs.append(record)
                print(f"{method} {order}-gram T={temp}, greedy={greedy}: {record['length']} 词; Top-1={result['top1']:.3f}", flush=True)
            del scorer
            gc.collect()
        (FULL / "ranking.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        (FULL / "generation.json").write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    db.close()


if __name__ == "__main__":
    main()

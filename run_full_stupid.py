"""在完整训练集上用磁盘计数构建 1—10 阶 Stupid Backoff。"""

from __future__ import annotations

import argparse
import json
import sqlite3
import struct
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "results" / "completed" / "train.txt"
RESULT = ROOT / "results" / "full"
WORK = Path.home() / "hw1_stupid_full"
DB = WORK / "stupid_counts.sqlite3"
VOCAB = RESULT / "stupid_vocab.json"
MANIFEST = RESULT / "stupid.json"
RESULT.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)


def save(value):
    MANIFEST.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_vocab():
    if VOCAB.exists():
        return json.loads(VOCAB.read_text(encoding="utf-8"))
    counts = Counter()
    sentences = 0
    with TRAIN.open("r", encoding="utf-8") as file:
        for line in file:
            counts.update(line.split())
            sentences += 1
    words = ["<s>", "</s>"] + sorted(counts)
    value = {"words": words, "sentences": sentences, "tokens": sum(counts.values())}
    VOCAB.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return value


def connection():
    db = sqlite3.connect(DB)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA temp_store=MEMORY")
    db.execute("PRAGMA cache_size=-250000")
    return db


def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i+size]


def build_order(db, order, ids):
    db.execute(f"DROP TABLE IF EXISTS counts_{order}")
    db.execute(f"CREATE TABLE counts_{order} (key BLOB PRIMARY KEY, c INTEGER NOT NULL) WITHOUT ROWID")
    db.commit()
    template = struct.Struct("<" + "I" * order)
    sql = f"INSERT INTO counts_{order}(key,c) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET c=c+excluded.c"
    batch = Counter()
    lines = 0
    tokens = 0
    begin = time.perf_counter()
    with TRAIN.open("r", encoding="utf-8") as file:
        for line in file:
            words = [ids[x] for x in line.split()]
            seq = [0] * 9 + words + [1]
            for index in range(9, len(seq)):
                batch[template.pack(*seq[index-order+1:index+1])] += 1
            lines += 1
            tokens += len(words)
            if lines % 500 == 0:
                db.executemany(sql, batch.items())
                db.commit()
                batch.clear()
            if lines % 25000 == 0:
                print(f"Stupid {order}-gram: {lines} / 227527 句", flush=True)
    if batch:
        db.executemany(sql, batch.items())
        db.commit()
    elapsed = time.perf_counter() - begin
    count = db.execute(f"SELECT count(*) FROM counts_{order}").fetchone()[0]
    return {"method": "Stupid Backoff", "order": order, "train_seconds": elapsed, "sentences": lines, "tokens": tokens, "unique_ngrams": count, "storage": str(DB), "alpha": 0.4}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", type=int, nargs="*", default=list(range(1, 11)))
    args = parser.parse_args()
    vocab = prepare_vocab()
    ids = {word: i for i, word in enumerate(vocab["words"])}
    data = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"corpus": "complete processed train split", "sentences": vocab["sentences"], "tokens": vocab["tokens"], "rows": []}
    completed = {x["order"] for x in data["rows"]}
    db = connection()
    for order in args.orders:
        if order in completed:
            print(f"Stupid {order}-gram 已完成，跳过", flush=True)
            continue
        row = build_order(db, order, ids)
        data["rows"].append(row)
        data["rows"].sort(key=lambda x:x["order"])
        save(data)
        print(f"Stupid {order}-gram: {row['train_seconds']:.1f}s, {row['unique_ngrams']:,} 个不同组合", flush=True)
    db.close()
    print(f"产物：{MANIFEST}，计数库：{DB}", flush=True)


if __name__ == "__main__":
    main()

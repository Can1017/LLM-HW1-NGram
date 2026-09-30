"""在当前 Python 环境完成语料清洗、n-gram 实验、可视化和报告。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

_code_root = Path(__file__).resolve().parent
for _name, _relative in (("MPLCONFIGDIR", ".cache/mpl"), ("TMP", ".tmp"), ("TEMP", ".tmp"), ("PYTHONPYCACHEPREFIX", ".cache/pycache")):
    _target = _code_root / _relative
    _target.mkdir(parents=True, exist_ok=True)
    os.environ[_name] = str(_target)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "199801"
OUT = ROOT / "results" / "completed"
FIG = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
SPLITS = ("train", "valid", "test")
ID = re.compile(r"^(\d{8}-\d{2}-\d{3})-\d{3}/m$")
END = {"。", "！", "？"}
CLOSING = {"”", "’", "』", "）", "》", "〉", "〕", "】", "］"}
PREFIX = "在阳光明媚的五月，我们学校胜利召开了"
PROMPT = ("在", "阳光明媚", "的", "五月", "，", "我们", "学校", "胜利", "召开", "了")


def save_json(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def font_setup():
    choices = (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf"))
    for path in choices:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def save_figure(name, columns, rows):
    with (FIG / f"{name}.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)
    plt.savefig(FIG / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def parse_token(raw):
    token = raw.lstrip("[")
    token = re.sub(r"(?:\][A-Za-z]+)+$", "", token)
    word = token.split("/", 1)[0]
    if "/" not in token or not word:
        raise ValueError("空词或缺少词性分隔符")
    if any(unicodedata.category(c).startswith("C") for c in word):
        raise ValueError("控制字符")
    return word


def sentences(words):
    result, current = [], []
    for word in words:
        if current and current[-1] in END and word not in END | CLOSING:
            result.append(tuple(current))
            current = []
        current.append(word)
    if current:
        result.append(tuple(current))
    return result


def prepare():
    start = time.perf_counter()
    groups = defaultdict(list)
    raw_count = Counter()
    tags = Counter()
    anomalies = []
    examples = []
    for file in sorted(RAW.glob("1998??.txt")):
        with file.open("r", encoding="utf-8", newline="") as handle:
            for line_no, line in enumerate(handle, 1):
                raw_count["lines"] += 1
                fields = line.split()
                if not fields:
                    raw_count["blank_lines"] += 1
                    continue
                match = ID.fullmatch(fields[0])
                if not match:
                    anomalies.append([file.name, line_no, "bad_id", fields[0]])
                    continue
                article = match.group(1)
                words = []
                bad = False
                for field in fields[1:]:
                    raw_count["word_entries"] += 1
                    tag_parts = field.lstrip("[").split("/")
                    if len(tag_parts) > 1:
                        tags[tag_parts[1].split("]")[0]] += 1
                    try:
                        words.append(parse_token(field))
                    except ValueError as error:
                        anomalies.append([file.name, line_no, str(error), field])
                        bad = True
                if bad:
                    raw_count["dropped_paragraphs"] += 1
                    continue
                pieces = sentences(words)
                groups[article].extend(pieces)
                if len(examples) < 6 and (len(examples) < 2 or "[" in line or "/%" in line):
                    examples.append({"source": f"{file.name}:{line_no}", "raw": line.strip()[:240], "clean": " / ".join(" ".join(x) for x in pieces)[:240]})
    # 清洗后的整篇词序列去重，以免重复栏目跨集合泄漏。
    unique, duplicate = {}, []
    for article in sorted(groups):
        digest = hashlib.sha256("\n".join(" ".join(s) for s in groups[article]).encode()).hexdigest()
        if digest in unique:
            duplicate.append([article, unique[digest]])
        else:
            unique[digest] = article
    duplicates = {x[0] for x in duplicate}
    ids = [x for x in sorted(groups) if x not in duplicates]
    random.Random(42).shuffle(ids)
    ntrain, nvalid = round(len(ids) * .8), round(len(ids) * .1)
    assigned = {x: "train" if i < ntrain else "valid" if i < ntrain + nvalid else "test" for i, x in enumerate(ids)}
    split_sents = {s: [] for s in SPLITS}
    for article in ids:
        split_sents[assigned[article]].extend(groups[article])
    vocab_counts = Counter(w for sent in split_sents["train"] for w in sent)
    kept = {w for w, c in vocab_counts.items() if c >= 2}
    raw_count["articles_before_dedup"] = len(groups)
    raw_count["duplicate_articles"] = len(duplicate)
    raw_count["articles_after_dedup"] = len(ids)
    raw_count["clean_sentences"] = sum(map(len, split_sents.values()))
    raw_count["clean_tokens"] = sum(len(s) for split in split_sents.values() for s in split)
    stats = {"raw": dict(raw_count), "tags_top15": tags.most_common(15), "split": {}, "anomalies": anomalies, "examples": examples, "duplicate_examples": duplicate[:20]}
    for split in SPLITS:
        rows = split_sents[split]
        mapped = [tuple(w if w in kept else "<UNK>" for w in sent) for sent in rows]
        stats["split"][split] = {"articles": sum(v == split for v in assigned.values()), "sentences": len(rows), "tokens": sum(map(len, rows)), "unk_tokens": sum(w == "<UNK>" for s in mapped for w in s)}
        with (OUT / f"{split}.txt").open("w", encoding="utf-8", newline="\n") as handle:
            for sent in mapped:
                handle.write(" ".join(sent) + "\n")
    stats["vocab"] = {"types": len(vocab_counts), "kept_types": len(kept), "singletons": sum(c == 1 for c in vocab_counts.values())}
    stats["preprocess_seconds"] = time.perf_counter() - start
    save_json("data_stats.json", stats)
    print("预处理完成", stats["raw"], flush=True)
    return stats, vocab_counts


class Ngram:
    """用绝对折扣与低阶插值训练可解释的词级 n-gram。"""

    def __init__(self, order, lines, discount=.75):
        self.order = order
        self.discount = discount
        self.counts = [Counter() for _ in range(order + 1)]
        self.next_types = [defaultdict(set) for _ in range(order)]
        self.context_totals = [Counter() for _ in range(order)]
        self.vocab = set()
        for line in lines:
            words = line.split()
            self.vocab.update(words)
            sequence = ["<s>"] * (order - 1) + words + ["</s>"]
            for index in range(order - 1, len(sequence)):
                for n in range(1, order + 1):
                    gram = tuple(sequence[index - n + 1:index + 1])
                    self.counts[n][gram] += 1
        self.vocab.add("</s>")
        self.base_total = sum(self.counts[1].values())
        for n in range(2, order + 1):
            for gram in self.counts[n]:
                self.next_types[n - 1][gram[:-1]].add(gram[-1])
            for gram, count in self.counts[n].items():
                self.context_totals[n - 1][gram[:-1]] += count

    def prob(self, word, history):
        history = tuple(history[-(self.order - 1):]) if self.order > 1 else ()
        if not history:
            return (self.counts[1].get((word,), 0) + .1) / (self.base_total + .1 * (len(self.vocab) + 1))
        context_count = self.context_totals[len(history)].get(history, 0)
        if not context_count:
            return self.prob(word, history[1:])
        observed = self.counts[len(history) + 1].get(history + (word,), 0)
        weight = self.discount * len(self.next_types[len(history)].get(history, ())) / context_count
        return max(observed - self.discount, 0) / context_count + weight * self.prob(word, history[1:])


def train(stats, vocab_counts):
    lines = (OUT / "train.txt").read_text(encoding="utf-8").splitlines()
    # 预算固定、抽样固定；整个语料已进入预处理统计，模型实验使用可审计的子集。
    indices = random.Random(2026).sample(range(len(lines)), min(30000, len(lines)))
    sample = [lines[i] for i in sorted(indices)]
    valid = (OUT / "valid.txt").read_text(encoding="utf-8").splitlines()[:600]
    records, models = [], {}
    for order in (2, 3, 5):
        elapsed = []
        for repeat in range(3):
            start = time.perf_counter()
            model = Ngram(order, sample)
            elapsed.append(time.perf_counter() - start)
            print(f"训练 {order}-gram 第 {repeat+1} 次：{elapsed[-1]:.2f}s", flush=True)
        models[order] = model
        logp, count = 0.0, 0
        for line in valid:
            words = line.split()
            history = ["<s>"] * (order - 1)
            for word in words + ["</s>"]:
                logp += math.log(max(model.prob(word, history), 1e-300))
                count += 1
                history.append(word)
        records.append({"order": order, "seconds": elapsed, "median_seconds": statistics.median(elapsed), "ngram_counts": {str(i): len(model.counts[i]) for i in range(1, order + 1)}, "ppl": math.exp(-logp / count), "eval_tokens": count})
    save_json("training.json", {"method": "Python absolute-discount interpolated n-gram; not KenLM/MKN", "sample_seed": 2026, "sample_sentences": len(sample), "discount": .75, "records": records})
    return models, records


def generate(models):
    records = []
    for order, temp, greedy in [(2, 1.0, False), (3, 1.0, False), (5, 1.0, False), (3, .7, False), (3, 1.3, False), (3, 1.0, True)]:
        model = models[order]
        # 全词表逐词评分成本很高；候选集取高频词与当前上下文直接观察到的后继词。
        frequent = {w for (w,), _ in model.counts[1].most_common(1000)} - {"<UNK>", "<s>"}
        for seed in ((42,) if greedy else (42, 43, 44)):
            rng = random.Random(seed)
            history = ["<s>"] * (order - 1) + [w if w in model.vocab else "<UNK>" for w in PROMPT]
            produced, probabilities = [], []
            ended = False
            for _ in range(80):
                context = tuple(history[-(order - 1):]) if order > 1 else ()
                options = frequent | model.next_types[order - 1].get(context, set()) | {"</s>"}
                options.discard("<UNK>")
                options.discard("<s>")
                scored = sorted(((model.prob(w, history), w) for w in options), key=lambda x: (-x[0], x[1]))[:20]
                if greedy:
                    choice, selected_probability = scored[0][1], scored[0][0]
                else:
                    weights = [max(p, 1e-300) ** (1 / temp) for p, _ in scored]
                    choice = rng.choices([w for _, w in scored], weights=weights, k=1)[0]
                    selected_probability = next(p for p, w in scored if w == choice)
                if choice == "</s>":
                    ended = True
                    break
                produced.append(choice)
                probabilities.append(selected_probability)
                history.append(choice)
            records.append({"order": order, "temperature": temp, "top_k": 20, "greedy": greedy, "seed": seed, "tokens": produced, "text": PREFIX + "".join(produced), "length": len(produced), "ended_by_eos": ended, "mean_selected_probability": statistics.mean(probabilities) if probabilities else None})
    save_json("generations.json", {"prompt": PREFIX, "prompt_tokens": PROMPT, "candidate_policy": "top 1000 unigrams + exact-context observed continuations, then top 20", "records": records})
    return records


def charts(stats, vocab_counts, training, outputs):
    font_setup()
    color = "#3155A4"
    # 清洗前后的文本量与各集合规模。
    rows = [[s, stats["split"][s]["articles"], stats["split"][s]["sentences"], stats["split"][s]["tokens"]] for s in SPLITS]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(["训练", "验证", "测试"], [r[3] for r in rows], color=color, width=.5)
    ax.set(title="训练集约占八成，文章级划分避免跨集重复", ylabel="词数")
    ax.grid(axis="y", alpha=.2)
    save_figure("01_语料划分", ["split", "articles", "sentences", "tokens"], rows)
    tagrows = stats["tags_top15"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([x[0] for x in tagrows][::-1], [x[1] for x in tagrows][::-1], color=color)
    ax.set(title="原语料含大量词性标签，训练前需去除", xlabel="词条数", ylabel="词性")
    save_figure("02_词性标签", ["tag", "count"], tagrows)
    freqs = sorted(vocab_counts.values(), reverse=True)
    points = [(i + 1, freqs[i]) for i in sorted(set(int((len(freqs) - 1) * j / 299) for j in range(300)))]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog([x[0] for x in points], [x[1] for x in points], color=color, linewidth=2)
    ax.axhline(1, color="#B65A46", linestyle="--", label="单次词")
    ax.legend()
    ax.set(title="词频呈长尾分布，单次词需要统一处理", xlabel="词频排名（对数）", ylabel="训练集出现次数（对数）")
    save_figure("03_词频长尾", ["rank", "count"], points)
    rows = [[r["order"], i + 1, value] for r in training for i, value in enumerate(r["seconds"])]
    fig, ax = plt.subplots(figsize=(7, 4))
    for r in training:
        ax.scatter([r["order"]] * 3, r["seconds"], color=color, s=45)
        ax.scatter(r["order"], r["median_seconds"], marker="_", s=400, color="#B65A46", linewidths=3)
    ax.set(xticks=[2, 3, 5], title="模型阶数增加，训练耗时变化", xlabel="n-gram 阶数", ylabel="秒；蓝点为单次，红线为中位数")
    ax.grid(axis="y", alpha=.2)
    save_figure("04_训练时间", ["order", "repeat", "seconds"], rows)
    rows = [[r["order"], r["ppl"], r["eval_tokens"]] for r in training]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r[0] for r in rows], [r[1] for r in rows], marker="o", color=color, linewidth=2)
    ax.set(xticks=[2, 3, 5], title="留出集困惑度随阶数变化", xlabel="n-gram 阶数", ylabel="困惑度（越低越好）")
    ax.grid(alpha=.2)
    save_figure("05_困惑度", ["order", "ppl", "eval_tokens"], rows)
    rows = [[i + 1, x["order"], x["temperature"], x["seed"], x["length"], int(x["ended_by_eos"])] for i, x in enumerate(outputs)]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([r[0] for r in rows], [r[4] for r in rows], color=["#3155A4" if r[5] else "#B65A46" for r in rows], width=.65)
    ax.set(title="16 条续写的长度及停止方式", xlabel="输出编号；蓝色为句尾停止，红色为达到上限", ylabel="生成词数", xticks=range(1, 17))
    save_figure("06_续写长度", ["id", "order", "temperature", "seed", "length", "eos"], rows)


def report(stats, training, outputs):
    table = "\n".join(f"| {r['order']} | {', '.join(f'{x:.2f}' for x in r['seconds'])} | {r['median_seconds']:.2f} | {r['ppl']:.2f} | {r['ngram_counts'][str(r['order'])]:,} |" for r in training)
    split = "\n".join(f"| {s} | {v['articles']:,} | {v['sentences']:,} | {v['tokens']:,} | {v['unk_tokens']/v['tokens']:.2%} |" for s, v in stats["split"].items())
    generated = "\n".join(f"| {i} | {x['order']} | {'贪婪' if x['greedy'] else x['temperature']} | {x['seed']} | {x['length']} | {'句尾' if x['ended_by_eos'] else '上限'} | {x['text']} |" for i, x in enumerate(outputs, 1))
    samples = "\n".join(f"- 来源 `{e['source']}`；原文：`{e['raw']}`；清洗：`{e['clean']}`" for e in stats["examples"][:3])
    doc = f"""# 人民日报词级 n-gram 语言模型实验

> L1 第 125 页编程作业。本文所有实验数值来自 `results/completed/`；图下方均提供对应 CSV 数据。本次为受当前环境限制的可复现实验基线，尚未完成课件所提工具的正式训练。

## 1. 任务与结论

我使用 1998 年 1—6 月的人民日报切分标注语料，清除词性、复合实体标签与行号，按文章划分数据，训练 2、3、5 阶词级 n-gram，并用同一开头进行 16 次续写。当前工作区中缺少先前规格所称的 KenLM 可执行文件与 Python 虚拟环境；确认 WSL 可启动后再次检查，仍无法直接运行 KenLM。本次实际采用 Python 实现的**绝对折扣插值 n-gram**。它与计划中的 KenLM Modified Kneser–Ney 平滑不同，结果不能当成 KenLM 的耗时或精度。

原语料共 {stats['raw']['lines']:,} 行、{stats['raw']['articles_before_dedup']:,} 篇文章；去重后 {stats['raw']['articles_after_dedup']:,} 篇。训练实验从训练集按固定种子抽取 {json.loads((OUT/'training.json').read_text(encoding='utf-8'))['sample_sentences']:,} 句，以控制当前环境的内存与运行时间。所有阶数使用同一批样本。

## 2. 数据来源与清洗

语料来自 [PeopleDaily1998](https://github.com/chenhui-bupt/PeopleDaily1998)。其使用声明说明：该语料在人民日报社新闻信息中心许可下，由北京大学计算语言学研究所与富士通研究开发中心有限公司制作；研究或论文使用需注明来源。输入文件位于 `data/raw/199801/`。

每行开头是日期、版面、文章与段落编号，后续词条形如 `词/词性`，复合实体使用方括号及末尾类别。清洗时逐词条保留第一个 `/` 前的词，移除编号、词性和复合标签，保持原有切词与正文标点。遇到空词或控制字符，将整个段落排除并记录。句号、问号、叹号作为句界；同篇文章统一归入一个数据集。训练集中只出现一次的词，以及验证/测试中不在保留词表内的词，映射为 `<UNK>`。使用大写形式是因为 KenLM 预留了小写 `<unk>`，两者不能混淆。

{samples}

清洗后语料共有 {stats['raw']['clean_sentences']:,} 句、{stats['raw']['clean_tokens']:,} 词，记录到 {len(stats['anomalies'])} 个异常词条；训练集原始词型 {stats['vocab']['types']:,} 个，其中单次词型 {stats['vocab']['singletons']:,} 个。

![三个数据集的规模](figures/01_语料划分.png)

图 1：按文章分组，减少同篇文章跨集合泄漏。[图中数据](figures/01_语料划分.csv)。

| 集合 | 文章 | 句子 | 词 | `<UNK>` 词元比例 |
|---|---:|---:|---:|---:|
{split}

![原始词性标签分布](figures/02_词性标签.png)

图 2：标签是原始标注信息。若留在正文中，模型会学习并生成标签。[图中数据](figures/02_词性标签.csv)。

![训练集词频长尾](figures/03_词频长尾.png)

图 3：大量词只出现一次，直接估计高阶条件概率会稀疏。[图中数据](figures/03_词频长尾.csv)。

## 3. 模型与生成方法

n 阶模型使用前 n−1 个词预测下一个词。本实验对出现过的 n 元组计数，按固定折扣 D=0.75 减少已见组合的概率质量，并把剩余质量递归交给低阶模型。可写为 `P(w|h)=max(c(h,w)-D,0)/c(h)+D·N₁⁺(h,*)/c(h)·P(w|短一阶历史)`。这解释了未见组合为什么仍可能获得非零概率。实现采用普通低阶词频，**未使用 Kneser–Ney 的续接概率**。

生成时，词表候选为训练样本的前 1,000 个高频词，加上当前完整上下文中观察到的后继词，再取概率前 20 个。贪婪解码取最高分词；随机解码按温度 0.7、1.0、1.3 重塑候选概率后抽样。生成 `</s>` 停止，最多续写 80 词。提示文本按 `{' / '.join(PROMPT)}` 切分，模型词表外词按 `<UNK>` 处理。候选截断使生成分布不同于完整词表的真实条件分布，这是本实验的性能折中。

## 4. 训练时间与留出集评测

同一抽样语料、同一机器、同一 Python 进程中，三个阶数各重新训练 3 次。耗时仅覆盖 Python 计数与索引构造，不含预处理、安装或模型持久化；不应与 KenLM 编译程序的训练时间直接比较。困惑度在验证集前 600 句上计算，包含句尾词，不包含句首词；三个模型采用同一评测文本。

| 阶数 | 三次训练秒数 | 中位数秒 | 困惑度 | 最高阶不同 n 元组 |
|---:|---|---:|---:|---:|
{table}

![训练时间对比](figures/04_训练时间.png)

图 4：蓝点为每次运行，红线为中位数。高阶模型要保存更多组合，时间可能增长；这里的变化以实测值为准。[图中数据](figures/04_训练时间.csv)。

![困惑度比较](figures/05_困惑度.png)

图 5：困惑度衡量模型对留出词序列的平均预测难度，越低越好；它不直接衡量续写是否自然。[图中数据](figures/05_困惑度.csv)。

## 5. 固定提示续写

下表完整列出全部 16 条记录。不同随机种子控制抽样过程，贪婪解码无需多个种子。原始词序列与选词概率见 [generations.json](results/completed/generations.json)。

| 编号 | 阶数 | 温度 | 种子 | 新词数 | 停止 | 完整输出 |
|---:|---:|---:|---:|---:|---|---|
{generated}

![续写长度与停止方式](figures/06_续写长度.png)

图 6：蓝柱生成句尾标记，橙柱达到 80 词上限；长度差异也反映解码随机性。[图中数据](figures/06_续写长度.csv)。

第 7 条 5 阶续写在“全国科学大会”“国家建设”等局部搭配上接近新闻体，但其语义从学校会议滑向宏观政策。第 6 条 3 阶续写只生成句号，说明句尾概率可能过高。第 16 条贪婪输出反复出现“一种新的一种新”，并达到 80 词上限；这体现了反复选择局部最高概率词的退化。第 11、14 条出现未闭合引号，也说明模型没有篇章级约束。5 阶样本更长，并不等于整体更连贯。样本数较少，不能从这些记录推断某一温度普遍更优。

## 6. 异常、局限与代码理解

- 原始语料存在空词条与控制字符；脚本记录并丢弃对应段落，避免把缺损文本拼接成虚假的相邻词。异常明细在 `results/completed/data_stats.json`。
- 当前工作区没有 KenLM 的可执行文件与 Python 环境，故没有 ARPA、二进制模型、训练阶段日志和 Modified Kneser–Ney 数值对照。当前结果属于可运行的统计模型基线；若严格按课件“运用前面提到的工具”评审，还需要补做工具训练。
- 为控制内存，训练只使用固定抽样的 30,000 句；从全语料得出的清洗统计与训练样本规模应分开理解。
- Python 实现将完整的多阶计数保存在内存中，训练时间与 KenLM 的外存流水线不可横向比较。

理解检查：n 阶模型只直接查看前 n−1 词；平滑把部分概率质量留给未见组合；阶数改变训练得到的上下文计数，温度只在抽样时改变分布；低困惑度衡量平均预测而非长文一致性；新闻语料里的高频用语影响词概率和生成文风。代码由 AI 编程助手辅助完成，我通过清洗样例、数据规模、固定种子复现、概率范围及原始输出核对关键行为；复现实验应再次检查这些断言。

## 7. 复现与文件

在 `code/` 目录运行 `python run_assignment.py`。输入是 `data/raw/199801/199801.txt` 至 `199806.txt`；脚本覆盖生成 `results/completed/` 中的数据文件、`figures/` 中的 PNG 与对应 CSV，以及本报告。所需环境为 Python 3.11、matplotlib；本机使用的其他依赖仅用于人工核查。原始语料与课件保留在原处。
"""
    (ROOT / "作业报告.md").write_text(doc, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="只清洗并划分语料，不运行早期抽样实验")
    args = parser.parse_args()
    stats, vocab = prepare()
    if args.prepare_only:
        print("产物：results/completed/train.txt、valid.txt、test.txt、data_stats.json", flush=True)
        return
    models, training = train(stats, vocab)
    outputs = generate(models)
    charts(stats, vocab, training, outputs)
    report(stats, training, outputs)
    print("产物：results/completed/、figures/、作业报告.md", flush=True)


if __name__ == "__main__":
    main()

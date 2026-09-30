"""根据全量实验的 JSON、日志及源码生成正式作业报告与可核对图表。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
FULL = ROOT / "results" / "full"
FIG = ROOT / "figures" / "full"
FIG.mkdir(parents=True, exist_ok=True)
stats = json.loads((ROOT / "results/completed/data_stats.json").read_text(encoding="utf-8"))
ken = json.loads((FULL / "kenlm.json").read_text(encoding="utf-8"))["rows"]
stupid = json.loads((FULL / "stupid.json").read_text(encoding="utf-8"))["rows"]
ranking = json.loads((FULL / "ranking.json").read_text(encoding="utf-8"))
generation = json.loads((FULL / "generation.json").read_text(encoding="utf-8"))

font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
plt.rcParams.update({"font.family": font_manager.FontProperties(fname="C:/Windows/Fonts/msyh.ttc").get_name(),
                     "axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})
BLUE, RED, GRAY = "#244b91", "#c65d42", "#657185"


def save(name, headers, rows):
    with (FIG / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(headers)
        writer.writerows(rows)
    plt.savefig(FIG / f"{name}.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close()


def charts():
    split = stats["split"]
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ["训练", "验证", "测试"]
    values = [split[k]["tokens"] / 1e6 for k in ("train", "valid", "test")]
    bars = ax.bar(labels, values, color=[BLUE, GRAY, GRAY], width=.48)
    ax.bar_label(bars, fmt="%.3f", padding=4)
    ax.set(ylabel="词数（百万）", title="完整语料按文章划分；训练使用全部训练集")
    ax.set_ylim(0, max(values) * 1.17)
    save("01_语料规模", ["split", "articles", "sentences", "tokens", "unknown_tokens"],
         [[k, *[split[k][x] for x in ("articles", "sentences", "tokens", "unk_tokens")]] for k in ("train", "valid", "test")])

    orders = [r["order"] for r in ken]
    cumulative = []
    running = 0
    for r in stupid:
        running += r["train_seconds"]
        cumulative.append(running)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), layout="constrained")
    axes[0].plot(orders, [r["train_seconds"] for r in ken], "o-", color=BLUE, lw=2)
    axes[0].annotate(f"{ken[-1]['train_seconds']:.1f} s", (10, ken[-1]["train_seconds"]), xytext=(-60, -26), textcoords="offset points", arrowprops={"arrowstyle": "->"})
    axes[0].set(title="KenLM 独立训练", ylabel="秒", xlabel="n-gram 阶数", xticks=orders)
    axes[1].plot(orders, cumulative, "s-", color=RED, lw=2)
    axes[1].annotate(f"{cumulative[-1]:.1f} s", (10, cumulative[-1]), xytext=(-65, -27), textcoords="offset points", arrowprops={"arrowstyle": "->"})
    axes[1].set(title="Stupid Backoff 累计计数", xlabel="n-gram 阶数", xticks=orders)
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("完整训练集上的 1—10 阶计算成本；两图纵轴范围不同")
    save("02_全量训练耗时", ["order", "kenlm_independent_seconds", "stupid_incremental_seconds", "stupid_cumulative_seconds"],
         [[n, ken[n-1]["train_seconds"], stupid[n-1]["train_seconds"], cumulative[n-1]] for n in orders])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), layout="constrained")
    ppl = [r["evaluation"]["test"]["ppl"] for r in ken]
    axes[0].plot(orders, ppl, "o-", color=BLUE, lw=2)
    axes[0].set(xlabel="n-gram 阶数", ylabel="perplexity（对数轴）", yscale="log", xticks=orders, title="1—10 阶全貌")
    axes[1].plot(orders[2:], ppl[2:], "o-", color=BLUE, lw=2)
    axes[1].set(xlabel="n-gram 阶数", ylabel="perplexity", xticks=orders[2:], title="3—10 阶局部放大")
    for n in (3, 5, 6, 10):
        axes[1].annotate(f"{ppl[n-1]:.1f}", (n, ppl[n-1]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle("完整测试集 perplexity：6 阶后改善趋于饱和")
    save("03_全测试集困惑度", ["order", "valid_ppl", "test_ppl", "test_nll_nats_per_token"],
         [[n, ken[n-1]["evaluation"]["valid"]["ppl"], ppl[n-1], ken[n-1]["evaluation"]["test"]["nll_nats_per_token"]] for n in orders])

    fig, ax = plt.subplots(figsize=(8.7, 4.4))
    sizes = [r["arpa_bytes"] / 1048576 for r in ken]
    ax.plot(orders, sizes, "o-", color=RED, lw=2)
    ax.annotate(f"{sizes[-1]:.0f} MiB", (10, sizes[-1]), xytext=(-85, 17), textcoords="offset points", arrowprops={"arrowstyle": "->"})
    ax.set(xlabel="n-gram 阶数", ylabel="ARPA 文件大小（MiB）", xticks=orders, title="模型阶数增加带来的存储开销")
    ax.grid(alpha=.2)
    save("04_模型大小", ["order", "arpa_bytes", "peak_rss_kb", "highest_order_ngrams"],
         [[n, ken[n-1]["arpa_bytes"], ken[n-1]["peak_kb"], ken[n-1]["ngram_counts"].get(str(n), 0)] for n in orders])

    fig, ax = plt.subplots(figsize=(8.7, 4.4))
    for method, color in (("KenLM", BLUE), ("Stupid Backoff", RED)):
        data = [r for r in ranking if r["method"] == method]
        ax.plot([r["order"] for r in data], [100 * r["top1"] for r in data], "o-", color=color, lw=2, label=method)
    ax.set(xlabel="n-gram 阶数", ylabel="Top-1 命中率（%）", xticks=orders,
           title="相同测试位置与受限候选集上的下一词排序")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    save("05_受限候选排序", ["method", "order", "positions", "top1", "top5", "mrr"],
         [[r[k] for k in ("method", "order", "positions", "top1", "top5", "mrr")] for r in ranking])

    base = [r for r in generation if r["temperature"] == 1 and not r["greedy"]]
    fig, ax = plt.subplots(figsize=(8.7, 4.4))
    for method, color, offset in (("KenLM", BLUE, -.16), ("Stupid Backoff", RED, .16)):
        data = [r for r in base if r["method"] == method]
        ax.bar([r["order"] + offset for r in data], [r["length"] for r in data], width=.31, color=color, label=method)
    ax.axhline(80, color=GRAY, ls="--", lw=1, label="长度上限")
    ax.set(xlabel="n-gram 阶数", ylabel="生成新词数", xticks=orders, title="固定提示、温度 1.0、top-k 20、种子 42 的续写长度")
    ax.legend(frameon=False, ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=.2)
    save("06_续写长度", ["method", "order", "length", "ended_by_eos", "seed"],
         [[r["method"], r["order"], r["length"], int(r["eos"]), r["seed"]] for r in base])

    phases = []
    names = {1: "计数和排序", 2: "调整计数", 3: "初始概率", 4: "插值概率", 5: "写入 ARPA"}
    for n in (3, 6, 10):
        record = json.loads((ROOT / "logs" / "full" / f"kenlm_{n}.phases.json").read_text(encoding="utf-8"))
        for left, right in zip(record, record[1:]):
            phases.append([n, left["stage"], names[left["stage"]], right["start_seconds"] - left["start_seconds"]])
    fig, ax = plt.subplots(figsize=(8.7, 4.3))
    colors = ["#244b91", "#4f71af", "#869bcc", "#bbc8df", "#c65d42"]
    for n in (3, 6, 10):
        bottom = 0
        for stage in range(1, 6):
            duration = next(r[3] for r in phases if r[0] == n and r[1] == stage)
            ax.bar(str(n), duration, bottom=bottom, width=.48, color=colors[stage-1], label=names[stage] if n == 3 else None)
            bottom += duration
    ax.set(title="KenLM 训练过程：10 阶 ARPA 写入占用较多时间", xlabel="n-gram 阶数", ylabel="阶段耗时（秒）")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=.2)
    save("07_KenLM训练阶段", ["order", "stage", "stage_name", "seconds"], phases)


def f(v, digits=1):
    return f"{v:,.{digits}f}"


def build():
    split = stats["split"]
    table = ["| 阶数 | KenLM 训练 / s | Stupid 本阶 / s | Stupid 累计 / s | 测试 perplexity | ARPA / MiB |", "|---:|---:|---:|---:|---:|---:|"]
    accumulated = 0
    for k, s in zip(ken, stupid):
        accumulated += s["train_seconds"]
        table.append(f"| {k['order']} | {f(k['train_seconds'])} | {f(s['train_seconds'])} | {f(accumulated)} | {f(k['evaluation']['test']['ppl'])} | {f(k['arpa_bytes']/1048576)} |")
    generations = ["| 方法 | 阶数 | 生成新词 | 结束方式 | 续写节选 |", "|---|---:|---:|---|---|"]
    for r in generation:
        if r["temperature"] != 1 or r["greedy"]:
            continue
        excerpt = "".join(r["tokens"])[:65].replace("|", "｜")
        generations.append(f"| {r['method']} | {r['order']} | {r['length']} | {'句尾标记' if r['eos'] else '80 词上限'} | {excerpt} |")
    variants = ["| 方法 | 解码 | 新词数 | 续写节选 |", "|---|---|---:|---|"]
    for r in generation:
        if r["order"] != 5:
            continue
        label = "贪婪" if r["greedy"] else f"T={r['temperature']:.1f}, top-k=20"
        variants.append(f"| {r['method']} | {label} | {r['length']} | {''.join(r['tokens'])[:75].replace('|','｜')} |")
    best = min(ken, key=lambda r:r["evaluation"]["test"]["ppl"])
    doc = f"""# 人民日报分词语料的 1—10 阶 n-gram 建模与续写实验

> 课程编程作业技术报告 · L1 第 125 页 · 实验记录基于本机运行产物

## 摘要

本实验以 1998 年 1—6 月人民日报切分标注语料为对象，清除词性及文档标签，在文章级隔离的训练、验证、测试划分上建立词级 n-gram 模型。KenLM 的 Modified Kneser–Ney 模型与自行实现的 Stupid Backoff 计数模型均覆盖 1—10 阶；训练使用**全部 227,527 个训练句、5,790,057 个词**。KenLM 在完整测试集上计算 perplexity（困惑度），两个模型在同一测试位置比较候选词排序，并以固定提示、随机种子 42 完成续写。结果表明，测试集 perplexity 在 6 阶附近趋于稳定，继续增加阶数仍显著增加训练及模型存储成本。

## 1. 任务依据与实验边界

课件要求对人民日报切分标注版进行标签清理、n-gram 建模、训练时间比较、训练过程观察、不同参数续写及结果分享。提示文本为“在阳光明媚的五月，我们学校胜利召开了”。课件没有指定必须使用 1—10 阶，也没有要求对随机种子做对照；本报告将阶数扩展到 1—10，采样统一固定为 42，以增加阶数曲线的观察点，同时保持续写比较的可控性。

数据来自本地 1998 年 1—6 月人民日报分词标注文件，公开来源为 [PeopleDaily1998](https://github.com/chenhui-bupt/PeopleDaily1998)。语料使用声明说明，该语料由人民日报社新闻信息中心许可，北京大学计算语言学研究所与富士通研究开发中心有限公司共同制作。方法与工具来源列于文末参考文献和项目；所有表格、图和续写均由当前工程的运行结果生成。

## 2. 数据清洗与实验设计

原始语料共 {stats['raw']['lines']:,} 行、{stats['raw']['articles_before_dedup']:,} 篇文章。清洗时去除文章编号、词性后缀及复合实体外层标注，保留实体内部已有词界和正文标点。例如，`[中央/n 人民/n 广播/vn 电台/n]nt` 转为 `中央 人民 广播 电台`。这一处理防止 `n`、`v` 等标签被当作正文词训练；原文标点并非多余标签，不予删除。异常控制字符记录于 [原始数据统计](results/completed/data_stats.json)，不做无记录的静默丢弃。

按文章去除 {stats['raw']['duplicate_articles']} 篇重复记录后，保留 {stats['raw']['articles_after_dedup']:,} 篇、{stats['raw']['clean_sentences']:,} 句、{stats['raw']['clean_tokens']:,} 词。划分在文章级进行，词表只由训练集建立；训练中仅出现一次的词和评测集词表外词映射为 `<UNK>`。训练、验证、测试各包含 {split['train']['articles']:,}、{split['valid']['articles']:,}、{split['test']['articles']:,} 篇文章。训练数据是**全量训练划分**；验证与测试保留为独立留出集，不能并入训练。

<img src="./figures/full/01_语料规模.png" alt="图 1：语料划分" width="900">

*图 1　词级训练、验证、测试规模。*

训练端将每句转为一行、以空格分词，文章之间不拼接上下文。词首、词尾标记用于模型内部计数。`lmplz` 是 KenLM 的语言模型估计程序：它读取逐行分词语料，统计 n-gram，使用 Modified Kneser–Ney 平滑估计条件概率，并输出 ARPA 文本模型；这里的 `-o` 指模型阶数，`-S 2G` 限定排序内存，`-T` 指定临时目录。各阶训练统一使用 `--discount_fallback` 配置。Stupid Backoff 的回退系数为 0.4。KenLM 的概率经过归一化，可计算 perplexity；Stupid Backoff 的分数不构成规范概率分布，因此只比较同一候选集上的排序，**不计算其 perplexity**。

实验在 Windows 主机的 WSL Ubuntu 22.04 环境中完成，处理器为 Intel Core i5-12500H，WSL 可用内存约 7.6 GiB。KenLM 使用本地编译的 `lmplz` 与 Python 查询模块；SQLite 计数库放在 WSL 的 ext4 文件系统中，训练文本与 ARPA 先在该文件系统处理，再将结果复制回 `hw1/code`。这样避免 Windows 挂载目录上的内存映射故障；[环境脚本](env/ensure_full.sh) 和 [执行脚本](run_hw1.ps1) 记录了命令入口。

## 3. 关键实现与运行命令

清洗程序位于 [run_assignment.py](run_assignment.py)，KenLM 全量训练与评估位于 [run_full_kenlm.py](run_full_kenlm.py)，Stupid Backoff 磁盘计数位于 [run_full_stupid.py](run_full_stupid.py)，共同排序与续写位于 [run_full_compare.py](run_full_compare.py)。下面节选真实实现中的核心逻辑。n 阶模型以此前 n−1 个词为历史；未出现的组合由平滑或回退处理。

```python
# run_full_stupid.py：按阶计数，避免把全部高阶计数留在内存
seq = [0] * 9 + words + [1]  # 句首补齐，1 表示句尾
for index in range(9, len(seq)):
    batch[template.pack(*seq[index-order+1:index+1])] += 1
```

```python
# run_full_compare.py：观测到组合时用条件频率，否则递归回退
joint = self.count(context + (word,))
if joint:
    denominator = self.sentences if all(x == 0 for x in context) else self.count(context)
    return joint / denominator
return 0.4 * self.stupid(word, context[1:])
```

续写将各候选词的对数分数除以温度后做稳定 softmax，再从前 20 个候选词中抽样；生成 `</s>` 或达到 80 新词时停止。Stupid Backoff 的这一步只是在**受限候选词集合内归一化抽样权重**，不意味着其原始分数成为标准语言模型概率。提示词沿用原始语料的词级切分，所有实验只使用种子 42。

在 `hw1/code` 目录运行以下命令即可按现有依赖重建报告，或执行完整流水线：

```powershell
python build_full_report.py
.\\run_hw1.ps1
```

后一命令会重用已完成的步骤；首次构建 KenLM 与磁盘计数需较多磁盘空间。训练日志位于 [logs/full](logs/full)，本机生成的各阶 ARPA 模型位于 `models/full/`；模型文件体积较大，由运行命令重建。

## 4. 训练时间、过程与资源

计时口径：KenLM 为启动 `lmplz` 至 ARPA 写出完成，输入复制及评测不计入；Stupid Backoff 的“本阶”是该阶 SQLite 计数的构建时间，“累计”是从 1 阶开始获得所需各阶计数的时间。两者实现语言和存储结构不同，时间差不能解释为平滑算法本身的纯粹效率差异。下表为每阶一次正式运行的实测结果。

{chr(10).join(table)}

<img src="./figures/full/02_全量训练耗时.png" alt="图 2：训练时间" width="900">

*图 2　KenLM 独立训练与 Stupid Backoff 累计计数时间。*

KenLM 较低阶训练时间约 1—11 秒；完整训练集共 {split['train']['sentences']:,} 个句子，其 C++ 外部排序与计数实现使低阶模型训练较快。10 阶训练为 {f(ken[-1]['train_seconds'])} 秒。Stupid Backoff 逐阶处理相同的训练句，SQLite 随组合数增长而产生明显开销。各阶 KenLM 日志记录计数、调整计数、初始概率、插值概率及 ARPA 写出五阶段；可参见 [10 阶原始日志](logs/full/kenlm_10.stderr.log)。

<img src="./figures/full/07_KenLM训练阶段.png" alt="图 3：训练阶段" width="900">

*图 3　由日志时间戳计算的 3、6、10 阶阶段耗时。*

## 5. 测试集质量与阶数取舍

KenLM 在**完整测试集 {split['test']['sentences']:,} 句**上评分；每句包含句尾预测，合计 {ken[0]['evaluation']['test']['predicted_tokens']:,} 个预测位置。Perplexity（困惑度，记为 PPL）表示模型对实际测试词序列的平均预测难度：先计算各词条件概率的平均负对数，再取指数：

$$
\\mathrm{{PPL}}=\\exp\\left(-\\frac{{1}}{{N}}\\sum_{{i=1}}^{{N}}\\ln p(w_i\\mid h_i)\\right).
$$

式中 $N$ 包含句尾预测位置，$h_i$ 为前文。**在相同分词与测试集上，数值越低表示模型赋予真实后续词的平均概率越高**；该指标不能单独衡量长篇续写的语义连贯性。1 阶至 5 阶下降明显，6 阶之后接近平台；最低测试集 perplexity 为 {f(best['evaluation']['test']['ppl'])}。

<img src="./figures/full/03_全测试集困惑度.png" alt="图 4：测试集 perplexity" width="900">

*图 4　左图用对数刻度展示 1—10 阶，右图放大 3—10 阶；曲线仅连接真实测量点。*

<img src="./figures/full/04_模型大小.png" alt="图 5：模型大小" width="900">

*图 5　ARPA 文本模型大小随阶数增长。*

高阶上下文可以改善局部概率估计，但会增加稀疏组合及存储。模型质量提升与计算成本应联合判断。KenLM 构建时统一启用了折扣回退选项；高阶稀疏计数如触发回退，不应将其结果表述为完全无回退的标准折扣估计。

## 6. 同候选集排序与文本续写

排序评测用种子 42 从长度至少为 3 词的测试句中随机取 150 个预测位置，候选集为**训练集最高频 100 词加真实目标词**。这是计算量受控的受限候选集实验；Top-1、Top-5 和平均倒数排名不能当作完整词表预测准确率。两种方法使用相同位置和候选集，因此可比较排序行为。

<img src="./figures/full/05_受限候选排序.png" alt="图 6：受限候选排序" width="900">

*图 6　受限候选集 Top-1 命中率。*

续写的主对照固定为 `T=1.0`、`top-k=20`、种子 42，并将阶数从 1 改至 10。为单独观察解码参数，5 阶模型另比较温度 0.7、1.3 和贪婪解码。所有生成只考虑训练词表中最高频 200 词与句尾标记；其流畅性和终止率受候选集限制。下表为**实际输出的开头节选**，完整词序列见 [generation.json](results/full/generation.json)，不只保留主观上最好的样例。

{chr(10).join(generations)}

<img src="./figures/full/06_续写长度.png" alt="图 7：续写长度" width="900">

*图 7　固定提示下的生成长度及 80 词截断情况。*

5 阶参数对照如下，变量仅为解码方式或温度，训练模型与提示保持一致：

{chr(10).join(variants)}

温度调整仅影响已经得到的候选词分数如何转成采样权重；模型阶数影响条件概率所使用的历史长度。贪婪解码总是选择当前最高分候选词，随机采样允许更多变化。困惑度反映词级平均预测能力，不直接保证长段落主题连贯；新闻语料的高频词与体裁也会使生成文本带有新闻报道风格。

具体输出也暴露了限制：1 阶主要生成标点和常见虚词，缺少可理解的主题；6—10 阶在固定候选集和种子下产生相同续写，说明新增的长上下文没有在当前提示及候选词上带来可见差异；5 阶 KenLM 在 `T=0.7` 时只生成一个句号，温度升至 1.3 则生成 25 个新词，但出现“发展的同时”等重复搭配。因而不能从单条输出推断温度提高总能改善文本，也不能把低困惑度等同于高质量长文本。

## 7. 结论与复现限制

本实验完成了标签清理、完整训练划分上的 1—10 阶双方法建模、训练日志观察、完整测试集 KenLM perplexity、共同候选集排序，以及固定种子续写。数据行数、处理后词数和各阶日志使实验规模可追溯。测试集 perplexity 在较高阶趋于平缓，而文件体积和计算成本持续增加，因而阶数选择应依据任务预算与生成表现共同判断。

实验仍有边界：每阶正式训练一次，训练时间反映该次运行；Stupid Backoff 是未归一化分数；排序及生成采用受限候选集，相关结论适用于本次候选范围。

**复核入口：**[语料统计](results/completed/data_stats.json) · [KenLM 各阶指标](results/full/kenlm.json) · [Stupid Backoff 各阶计数](results/full/stupid.json) · [排序指标](results/full/ranking.json) · [全部续写](results/full/generation.json) · [全部图表及 CSV](figures/full)

## 参考文献与项目

[1] Brants T, Popat A C, Xu P, et al. [Large Language Models in Machine Translation](https://aclanthology.org/D07-1090/). EMNLP-CoNLL, 2007. Stupid Backoff 方法来源。

[2] Heafield K, Pouzyrevsky I, Clark J H, et al. [Scalable Modified Kneser-Ney Language Model Estimation](https://aclanthology.org/P13-2121/). ACL, 2013. KenLM 模型估计方法。

[3] [KenLM 官方项目及 `lmplz` 用法](https://github.com/kpu/kenlm). 本实验的模型训练与评分工具。

[4] [PeopleDaily1998 语料项目](https://github.com/chenhui-bupt/PeopleDaily1998). 本实验的数据来源；语料制作与许可信息见随数据提供的 `shengming.doc`。

[5] [LynxPeng/LLM-HomeWork_N-Gram](https://github.com/LynxPeng/LLM-HomeWork_N-Gram)；[Highsun/n-gram-LM](https://github.com/Highsun/n-gram-LM). 课程相关实现参考项目。
"""
    (ROOT / "作业报告.md").write_text(doc, encoding="utf-8")


if __name__ == "__main__":
    charts()
    build()
    print(ROOT / "作业报告.md")

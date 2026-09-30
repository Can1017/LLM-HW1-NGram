"""根据实验 JSON 生成双方法对比图和可审阅的 Markdown 报告。"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results" / "methods"
FIG = ROOT / "figures"
LOG = ROOT / "logs" / "methods"


def read(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def graphic(name, header, rows):
    with (FIG / f"{name}.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
    plt.savefig(FIG / f"{name}.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close()


def style():
    font = Path("C:/Windows/Fonts/msyh.ttc")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(), "axes.unicode_minus": False, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def plots(training, evaluation, generation):
    style()
    blue, orange = "#3155A4", "#C05A44"
    rows = [[r["method"], r["order"], i+1, x] for r in training for i, x in enumerate(r["seconds"])]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for method, offset, color in (("KenLM", -.13, blue), ("Stupid Backoff", .13, orange)):
        records = [r for r in training if r["method"] == method]
        ax.scatter([r["order"]+offset for r in records for _ in r["seconds"]], [x for r in records for x in r["seconds"]], color=color, alpha=.58, s=43, label=method)
        ax.plot([r["order"]+offset for r in records], [r["median_seconds"] for r in records], color=color, linewidth=2)
    ax.set(title="相同语料下，两种实现的训练时间随阶数增长", xlabel="n-gram 阶数", ylabel="训练耗时（秒）", xticks=[2,3,5])
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    graphic("07_双方法训练时间", ["method","order","repeat","seconds"], rows)

    stage_names = {1:"计数与排序",2:"调整计数",3:"初始概率",4:"插值概率",5:"写入 ARPA"}
    rows = []
    for order in (2,3,5):
        phases = json.loads((LOG / f"kenlm_{order}_3.phases.json").read_text(encoding="utf-8"))
        for left, right in zip(phases, phases[1:]):
            rows.append([order, left["stage"], stage_names[left["stage"]], right["start_seconds"]-left["start_seconds"]])
    fig, ax = plt.subplots(figsize=(8, 4.6))
    palette = ["#3155A4", "#5D77B9", "#8298C9", "#AAB8D8", "#C05A44"]
    for order in (2,3,5):
        bottom = 0
        for stage in range(1,6):
            duration = next(r[3] for r in rows if r[0] == order and r[1] == stage)
            ax.bar(str(order), duration, bottom=bottom, width=.5, color=palette[stage-1], label=stage_names[stage] if order == 2 else None)
            bottom += duration
    ax.set(title="KenLM 训练日志显示计数与写入占主要时间", xlabel="n-gram 阶数", ylabel="阶段耗时（秒）")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    graphic("08_KenLM训练阶段", ["order","stage","name","seconds"], rows)

    rows = [[x["method"], x["order"], x["top1"], x["top5"], x["mrr"]] for x in evaluation]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for method, color in (("KenLM", blue), ("Stupid Backoff", orange)):
        values = [r for r in rows if r[0] == method]
        ax.plot([r[1] for r in values], [r[2]*100 for r in values], marker="o", color=color, linewidth=2, label=method)
    ax.set(title="相同候选集上的下一词 Top-1 命中率", xlabel="n-gram 阶数", ylabel="Top-1 命中率（%）", xticks=[2,3,5])
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    graphic("09_下一词命中率", ["method","order","top1","top5","mrr"], rows)

    ken = [x for x in evaluation if x["method"] == "KenLM"]
    rows = [[x["order"], x["nll_nats_per_token"], x["ppl"]] for x in ken]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot([x[0] for x in rows], [x[1] for x in rows], color=blue, marker="o", linewidth=2)
    ax.set(title="KenLM 留出集平均负对数似然随阶数变化", xlabel="n-gram 阶数", ylabel="NLL（自然对数／词）", xticks=[2,3,5])
    ax.grid(alpha=.2)
    graphic("10_留出集NLL", ["order","nll_nats_per_token","ppl"], rows)

    rows = [[i+1, x["method"], x["order"], x["temperature"], x["seed"], x["length"], int(x["eos"])] for i,x in enumerate(generation)]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False, layout="constrained")
    for ax, method in zip(axes, ("KenLM", "Stupid Backoff")):
        chosen = [r for r in rows if r[1] == method]
        ax.bar(range(1, len(chosen)+1), [r[5] for r in chosen], color=[blue if r[6] else orange for r in chosen], width=.75)
        ax.set(ylabel="新词数", title=method, xticks=range(1,len(chosen)+1))
        ax.grid(axis="y", alpha=.15)
    axes[-1].set_xlabel("各方法内部的续写编号；橙色表示达到 80 词上限")
    graphic("11_双方法续写长度", ["id","method","order","temperature","seed","length","eos"], rows)


def make_report(training, evaluation, generation, env, stats):
    record = {(r["method"], r["order"]): r for r in training}
    metrics = {(r["method"], r["order"]): r for r in evaluation}
    table = "\n".join(f"| {method} | {order} | {record[(method,order)]['median_seconds']:.2f} | {', '.join(f'{s:.2f}' for s in record[(method,order)]['seconds'])} | {metrics[(method,order)]['top1']:.1%} | {metrics[(method,order)]['top5']:.1%} | {metrics[(method,order)]['mrr']:.3f} |" for method in ("KenLM","Stupid Backoff") for order in (2,3,5))
    sizes = "\n".join(f"| {order} | {record[('KenLM',order)]['ngram_counts'][f'ngram {order}']:,} | {record[('KenLM',order)]['arpa_bytes']/1024/1024:.1f} |" for order in (2,3,5))
    ken_eval = "\n".join(f"| {r['order']} | {r['nll_nats_per_token']:.3f} | {r['ppl']:.1f} |" for r in evaluation if r["method"] == "KenLM")
    outputs = "\n".join(f"| {i} | {r['method']} | {r['order']} | {'贪婪' if r['greedy'] else r['temperature']} | {r['seed']} | {r['length']} | {'句尾' if r['eos'] else '上限'} | {r['text'].replace('|','｜')} |" for i,r in enumerate(generation,1))
    split = "\n".join(f"| {name} | {value['articles']:,} | {value['sentences']:,} | {value['tokens']:,} |" for name,value in stats["split"].items())
    rank3 = sorted((r for r in evaluation if r["order"] == 3), key=lambda x:x["method"])
    train_sample = read("training.json")
    kenlm_log = (LOG / "kenlm_5_3.stderr.log").read_text(encoding="utf-8")
    peak_kb = int(re.search(r"RSSMax:(\d+) kB", kenlm_log).group(1))
    ken_3 = metrics[("KenLM", 3)]["top1"]
    stupid_3 = metrics[("Stupid Backoff", 3)]["top1"]
    start = """# 人民日报 n-gram 语言模型：KenLM 与 Stupid Backoff 对照实验

> L1 第 125 页编程作业。所有训练时间、指标、续写都来自本机运行的实验文件；图旁提供 CSV 原始数据。本文采用 **KenLM + Stupid Backoff 两种方法**，比较阶数与采样参数。

## 一、任务、方法和主要发现

这次作业要求清除人民日报切分标注语料中的标签，训练 n-gram 模型，比较训练时间，观察训练过程，并对“在阳光明媚的五月，我们学校胜利召开了……”进行不同参数的续写。我选择 KenLM 的 Modified Kneser–Ney（MKN）概率模型和固定系数回退的 Stupid Backoff。两者共用同一批词级训练句、相同的 2／3／5 阶和评测位置。

**核心区别：**KenLM 的 MKN 给出归一化的条件概率；Stupid Backoff 在高阶组合未见时乘 0.4 回退，得到的是排序分数，通常不构成归一化概率。因此两者用相同候选集的 Top-1／Top-5 命中率及 MRR 比较；困惑度只对 KenLM 报告。训练时间是“本机 C++ KenLM 实现”与“本机 Python Stupid Backoff 实现”的实际耗时，反映算法与实现语言的共同影响。

## 二、数据集来源与预处理

数据来自本地 `data/raw/199801/` 的 1998 年 1—6 月人民日报切分标注语料，其公开来源为 [PeopleDaily1998](https://github.com/chenhui-bupt/PeopleDaily1998)。语料使用声明注明：人民日报社新闻信息中心许可，北京大学计算语言学研究所与富士通研究开发中心有限公司制作。课程微信群版本目前不在工作区，无法证明两份数据逐字节相同；若教师指定了另一版本，需要重新校验。

原始词条形如 `词/词性`，首词条是文章与段落编号，复合实体还含方括号。清洗保留词、原有分词和标点，去掉编号、词性与复合标签。遇到空词条或控制字符，丢弃对应段落并记录。按文章去重和划分数据，训练集建立词表；仅出现一次的训练词以及验证集未知词用 `<UNK>` 表示。

"""
    start += f"原始数据共 {stats['raw']['lines']:,} 行、{stats['raw']['articles_before_dedup']:,} 篇文章；记录 {len(stats['anomalies'])} 个异常词条，去重后 {stats['raw']['articles_after_dedup']:,} 篇。六个月数据清洗后有 {stats['raw']['clean_sentences']:,} 句。正式双方法训练从训练集以种子 2026 固定抽取 **{train_sample['sample_sentences']:,} 句、{train_sample['sample_tokens']:,} 词**，两个方法使用完全相同的抽样文本；这不等同于在六个月全量语料上训练。\n\n"
    start += f"| 集合 | 文章数 | 句数 | 词数 |\n|---|---:|---:|---:|\n{split}\n\n"
    start += """例如，`[中央/n  人民/n  广播/vn  电台/n]nt` 清洗为 `中央 人民 广播 电台`。标签若进入训练，会被当成可预测词，扭曲词频并污染续写。原始样例、异常与更多统计见 [数据统计](results/completed/data_stats.json)。

![语料划分](figures/01_语料划分.png)

图 1 的训练／验证／测试集合按文章划分；双方法训练使用训练集中的固定抽样。[数据表](figures/01_语料划分.csv)

![词性标签分布](figures/02_词性标签.png)

图 2 展示了为什么必须清理标注：常见的 n、v、w 等标签在原文中出现大量次数。[数据表](figures/02_词性标签.csv)

![词频长尾](figures/03_词频长尾.png)

图 3 展示单次词与稀疏问题；这也是平滑和回退的动机。[数据表](figures/03_词频长尾.csv)

## 三、技术与实际实验环境

- **KenLM：**本地源码编译的 `lmplz`，默认 Modified Kneser–Ney 平滑。训练输入为每行一句，阶数 2／3／5，排序内存上限 `-S 2G`，不剪枝；输出 ARPA 模型。源码提交号见 `results/methods/environment.json`。
- **Stupid Backoff：**Python 标准库实现计数；已见 n 元组使用 `c(hw)/c(h)`，未见组合递归乘 `α=0.4` 回退，最低阶为语料一元频率。它不做概率质量归一化。
- **环境：**Windows 主机上的 WSL2 Ubuntu 22.04，Python """ + env["python"] + f"""；CPU 为 {env.get('cpu_model', '12th Gen Intel Core i5-12500H')}，{env['cpu_count']} 个逻辑 CPU，WSL 可见内存约 {int(env['memory'].split()[1])/1024/1024:.2f} GiB；本机 GPU 为 NVIDIA GeForce RTX 3050 4GB。两种 n-gram 工具都在 CPU 上执行计数与查询，GPU 未参与。无需 AutoDL，因为这批固定样本在本机可完成。
- **文件系统：**KenLM 的 mmap 临时文件在 WSL 的 Windows 挂载盘触发段错误；相同 1,000 句在 WSL ext4 成功。正式训练在本机 WSL ext4 临时目录运行，模型与日志保存回 `code/`。此差异已通过探针日志验证。
- **资源实测：**KenLM 5 阶第 3 次运行的 `RSSMax` 为 {peak_kb/1024:.1f} MiB（见原始日志），低于分配给 WSL 的可用内存；本机可完成该规模训练。

## 四、实验步骤与训练过程

1. 运行 `python run_assignment.py` 完成原始语料清洗、文章划分和基础统计。
2. 在 WSL 中构建 KenLM 命令行与 Python 查询模块。构建脚本位于 `env/`；运行 `bash env/run_methods.sh` 进行双方法实验。
3. 固定抽取训练集的 30,000 句；每种方法分别训练 2／3／5 阶，各重复 3 次。只计训练过程，抽样、复制文件及评测不进入训练计时。
4. KenLM 每次训练记录五阶段的开始时间和原始 stderr。Stupid Backoff 记录 Python 计数与索引构造耗时。模型评测使用验证集前 600 句；排名比较从这些句中固定抽取 150 个预测位置。
5. 每种方法与阶数在温度 1.0 下使用种子 42、43、44 生成；固定 3 阶再比较温度 0.7、1.3 与贪婪解码。最长 80 个新词，句尾标记提前终止。

![两种方法训练时间](figures/07_双方法训练时间.png)

图 4 的点是每次实测，线连各阶中位数。KenLM 的 C++ 与 Stupid Backoff 的 Python 实现不能用来单独证明某种平滑算法更快。[数据表](figures/07_双方法训练时间.csv)

![KenLM 训练阶段](figures/08_KenLM训练阶段.png)

图 5 来自 KenLM 第 3 次运行的阶段时间戳，显示计数、调整计数、计算概率和写出 ARPA 的实际过程。[数据表](figures/08_KenLM训练阶段.csv) [原始日志](logs/methods/kenlm_5_3.stderr.log)

**关于 loss 曲线：**n-gram 使用计数、折扣和回退估计，没有神经网络式 epoch、反向传播或训练 loss 轨迹。为直观呈现模型质量变化，图 7 使用同一验证集在不同阶数下的平均负对数似然（NLL），而不虚构训练 loss。

## 五、训练与评测结果

| 方法 | 阶数 | 训练中位数／秒 | 三次训练／秒 | Top-1 | Top-5 | MRR |
|---|---:|---:|---|---:|---:|---:|
"""
    start += table + "\n\n"
    start += f"| KenLM 阶数 | 最高阶 n-gram 数量 | ARPA 文件大小／MiB |\n|---:|---:|---:|\n{sizes}\n\n"
    start += f"3 阶时，KenLM 的 Top-1 为 {ken_3:.1%}，Stupid Backoff 为 {stupid_3:.1%}；两者相差 {abs(stupid_3-ken_3)*150:.0f} 个命中位置（共 150 个），样本量不足以据此认定一种方法普遍更准确。KenLM 的 3 阶训练中位数为 {record[('KenLM',3)]['median_seconds']:.2f} 秒，Stupid Backoff Python 版为 {record[('Stupid Backoff',3)]['median_seconds']:.2f} 秒；实现语言差异显著。\n\n"
    start += """排名评测的候选集为训练样本最高频的 300 个词加当前位置的真实词，两个方法相同。因此 Top-1／Top-5 是**受限候选集结果**，不可当作完整词表的真实准确率；MRR 是真实词排名倒数的均值。训练样本之外的验证词均按预处理时的 `<UNK>` 规则读取。详细数据在 [evaluation.json](results/methods/evaluation.json)。

![下一词命中率](figures/09_下一词命中率.png)

图 6 用同一组预测位置比较方法，避免把未归一化 Stupid Backoff 分数当成概率。[数据表](figures/09_下一词命中率.csv)

| KenLM 阶数 | 平均 NLL／词 | 困惑度 |
|---:|---:|---:|
"""
    start += ken_eval + "\n\n"
    start += """困惑度为 `exp(NLL)`，只适用于归一化概率模型。Stupid Backoff 没有直接可比的困惑度。

![留出集 NLL](figures/10_留出集NLL.png)

图 7 比较的是**不同阶数模型**在验证集上的 NLL，不是单个模型训练期间逐步下降的 loss。[数据表](figures/10_留出集NLL.csv)

## 六、指定开头的续写与参数对比

统一提示为“在阳光明媚的五月，我们学校胜利召开了……”，省略号不输入模型。提示按 `在 / 阳光明媚 / 的 / 五月 / ， / 我们 / 学校 / 胜利 / 召开 / 了` 切词。候选集合取训练样本高频 300 词及当前完整上下文已见后继词，再保留得分前 20 个；随机采样对这 20 项按温度重新归一化。两个方法采用相同候选策略与种子。所有输出都保留，包括不通顺和提前结束的案例。

![两种方法续写长度](figures/11_双方法续写长度.png)

图 8 展示全部 32 条输出的长度与停止方式。长输出不自动代表更连贯；到上限说明模型未生成句尾。[数据表](figures/11_双方法续写长度.csv)

| 编号 | 方法 | 阶数 | 温度 | 种子 | 新词数 | 停止 | 完整续写 |
|---:|---|---:|---:|---:|---:|---|---|
"""
    start += outputs + "\n\n"
    sample_ken = next(x for x in generation if x["method"] == "KenLM" and x["order"] == 3 and x["temperature"] == 1 and x["seed"] == 42 and not x["greedy"])
    sample_stupid = next(x for x in generation if x["method"] == "Stupid Backoff" and x["order"] == 3 and x["temperature"] == 1 and x["seed"] == 42 and not x["greedy"])
    start += f"在相同的 3 阶、温度 1.0、种子 42 下，KenLM 生成 {sample_ken['length']} 个新词，Stupid Backoff 生成 {sample_stupid['length']} 个。两条输出都偏向新闻报道用语，并从学校会议滑向其他主题。这是具体失败案例，说明局部 n-gram 搭配无法维持全文主题。\n\n"
    start += """逐词输出和全部配置在 [generation.json](results/methods/generation.json)。比较时应同时看局部搭配、整体语义、重复和句尾。新闻语料的固定搭配会把文本引向政策与时政报道；高阶历史更长，却仍无法保证跨句主题一致。温度改变采样随机性，不能修复语料偏向或长距离依赖。

## 七、理解、限制与复现

**代码理解：**n 阶模型直接使用前 `n−1` 个词。KenLM 通过折扣把概率质量分给未见组合，并在低阶使用续接信息；Stupid Backoff 对未见高阶组合直接乘固定系数退到低阶。阶数在训练和查询中决定最大历史，温度仅在生成时改写候选分数。困惑度只量化平均预测，不证明长文逻辑通顺。

**实验限制：**本次训练是六个月语料中固定抽取的 30,000 句；原语料与课程群文件未做逐字节比对；方法时间受 C++／Python 实现差异影响；排名评测只覆盖 150 个位置和受限词集；32 条生成不能支撑对总体写作质量的显著性结论。本机 GPU 未用于这些 CPU 主导的 n-gram 算法。报告如实保留 KenLM 在 DrvFs 上失败的探针与成功切换记录。

**复现：**在 Windows 的 `code/` 目录运行 `python run_assignment.py` 完成预处理；在 WSL Ubuntu 22.04 运行 `bash env/build_kenlm_minimal.sh`、`bash env/build_kenlm_python.sh`，然后运行 `bash env/run_methods.sh`。最后在 Windows 运行 `python build_final_report.py` 重新生成图和本文。原始数据、抽样文本、模型、训练日志、JSON 和图片分别在 `data/raw/`、`results/methods/`、`models/methods/`、`logs/methods/`、`figures/`。代码由 AI 编程助手辅助完成；通过固定种子、同源数据、原始日志、模型可加载和生成输出核对结果。

**方法参考：**[KenLM 官方实现与说明](https://github.com/kpu/kenlm)、[Heafield 等，Scalable Modified Kneser-Ney Language Model Estimation（ACL 2013）](https://aclanthology.org/P13-2121/)、[Brants 等，Large Language Models in Machine Translation（EMNLP-CoNLL 2007）](https://aclanthology.org/D07-1090/)。
"""
    (ROOT / "作业报告.md").write_text(start, encoding="utf-8")


def main():
    training = read("training.json")["trainings"]
    evaluation = read("evaluation.json")
    generation = read("generation.json")
    environment = read("environment.json")
    stats = json.loads((ROOT / "results" / "completed" / "data_stats.json").read_text(encoding="utf-8"))
    assert len(training) == 6 and len(evaluation) == 6 and len(generation) == 32
    plots(training, evaluation, generation)
    make_report(training, evaluation, generation, environment, stats)
    print("产物：作业报告.md、figures/07—11 PNG 与 CSV")


if __name__ == "__main__":
    main()

# HW1：n-gram 语言模型实验

[作业报告](作业报告.md) 是评审入口；正文包含作业要求、实验方法、7 张图表、1—10 阶实测结果、代码讲解、续写及参考资料。

从 GitHub 克隆时请带上 KenLM 子模块。在 Windows PowerShell 中进入本目录，运行一条命令即可下载并校验语料、完成实验、生成报告；已有结果会复用：

```powershell
git clone --recurse-submodules https://github.com/Can1017/LLM-HW1-NGram.git
cd LLM-HW1-NGram
python -m pip install -r requirements.txt
.\run_hw1.ps1
```

要求 Windows Python 3.11（安装 `requirements.txt` 中的依赖）、WSL Ubuntu 22.04，以及编译 KenLM 所需的基础 C++ 工具链。语料由 `fetch_corpus.py` 按固定提交下载并校验 SHA-256；原始归档、清洗文本与大型模型不保存在本仓库中。

在已有工作区中，可直接运行：

```powershell
.\run_hw1.ps1
```

脚本依次获取和清洗语料、检查 KenLM 环境，完成 1—10 阶全量 KenLM 训练、Stupid Backoff 计数、同候选集排序与续写，再生成图表和报告。已完成的训练按结果文件跳过。单独更新图表与报告可运行 `python build_full_report.py`。

正式实验的结构化数据位于 `results/full/`，KenLM ARPA 模型位于 `models/full/`，训练日志位于 `logs/full/`，图表与同名 CSV 位于 `figures/full/`。Stupid Backoff 的 SQLite 计数库位于 WSL 用户目录的 `~/hw1_stupid_full/stupid_counts.sqlite3`；Linux 文件系统用于避免 Windows 挂载目录上的内存映射问题。`run_assignment.py` 的语料清洗结果在 `results/completed/`。

`run_methods.py`、`build_final_report.py` 是早期 2／3／5 阶抽样实验的留存脚本，不构成当前报告的证据。正式执行以本 README、当前全量实验脚本和报告为准。

检查产物一致性：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:PYTHONPATH=(Resolve-Path '.').Path
python -m pytest tests -q
```

报告由 `build_full_report.py` 生成；修改正文后如需重复生成，应同步修改该脚本中的报告模板。

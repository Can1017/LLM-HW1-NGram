"""对正式 1—10 阶实验产物做独立一致性检查。"""

import csv
import json
import math
import re
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "results" / "full"
COMPLETE = ROOT / "results" / "completed"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_full_training_and_evaluation_artifacts():
    stats = read_json(COMPLETE / "data_stats.json")["split"]
    for split in ("train", "valid", "test"):
        with (COMPLETE / f"{split}.txt").open(encoding="utf-8") as handle:
            sentences = [line.split() for line in handle]
        assert len(sentences) == stats[split]["sentences"]
        assert sum(map(len, sentences)) == stats[split]["tokens"]
        assert sum(word == "<UNK>" for sent in sentences for word in sent) == stats[split]["unk_tokens"]

    kenlm = read_json(FULL / "kenlm.json")["rows"]
    stupid = read_json(FULL / "stupid.json")["rows"]
    assert [row["order"] for row in kenlm] == list(range(1, 11))
    assert [row["order"] for row in stupid] == list(range(1, 11))
    for k, s in zip(kenlm, stupid):
        assert k["train_seconds"] > 0 and s["train_seconds"] > 0
        assert s["sentences"] == stats["train"]["sentences"]
        assert s["tokens"] == stats["train"]["tokens"]
        assert s["unique_ngrams"] > 0
        model = ROOT / k["model"]
        assert model.stat().st_size == k["arpa_bytes"]
        assert (ROOT / k["log"]).is_file()
        for split in ("valid", "test"):
            metric = k["evaluation"][split]
            assert metric["sentences"] == stats[split]["sentences"]
            assert metric["predicted_tokens"] == stats[split]["tokens"] + stats[split]["sentences"]
            assert math.isclose(metric["ppl"], math.exp(metric["nll_nats_per_token"]), rel_tol=1e-12)


def test_ranking_generation_and_report_links():
    ranking = read_json(FULL / "ranking.json")
    generation = read_json(FULL / "generation.json")
    assert {(r["method"], r["order"]) for r in ranking} == {
        (method, n) for method in ("KenLM", "Stupid Backoff") for n in range(1, 11)
    }
    assert all(r["positions"] == 150 and 0 <= r["top1"] <= r["top5"] <= 1 for r in ranking)
    assert len(generation) == 26
    assert {r["seed"] for r in generation} == {42}
    for row in generation:
        assert len(row["tokens"]) == row["length"] <= 80
        assert row["text"] == "在阳光明媚的五月，我们学校胜利召开了" + "".join(row["tokens"])
        assert not (row["eos"] and row["length"] == 80)

    report = (ROOT / "作业报告.md").read_text(encoding="utf-8")
    assert "微信群" not in report and "loss" not in report.lower()
    assert "perplexity" in report.lower() and "lmplz" in report
    assert "## 参考文献与项目" in report
    for target in re.findall(r"!?\[[^]]*\]\(([^)]+)\)", report):
        if not target.startswith(("https://", "http://")):
            assert (ROOT / target).exists(), target
    embedded = re.findall(r'<img src="([^"]+)"', report)
    assert len(embedded) == 7
    expected_prefix = "/Can1017/LLM-HW1-NGram/main/figures/full/"
    for target in embedded:
        parsed = urlparse(target)
        assert parsed.scheme == "https" and parsed.netloc == "raw.githubusercontent.com"
        assert parsed.path.startswith(expected_prefix)
        assert (ROOT / "figures" / "full" / unquote(parsed.path.rsplit("/", 1)[-1])).is_file()
    with (ROOT / "figures" / "full" / "05_受限候选排序.csv").open(encoding="utf-8-sig", newline="") as file:
        assert len(list(csv.DictReader(file))) == len(ranking)

"""验证 Stupid Backoff 的计数、回退及实验产物的关键约束。"""

import json
from pathlib import Path

from run_methods import Stupid


def test_stupid_seen_and_unseen():
    model = Stupid(3, ["我 爱 学习", "我 爱 中文", "你 爱 学习"])
    assert model.score("学习", ["我", "爱"]) == 0.5
    assert model.score("中文", ["你", "爱"]) == 0.4 * model.score("中文", ["爱"])
    assert model.score("没有", ["我", "爱"]) == 0


def test_experiment_evidence():
    root = Path(__file__).resolve().parents[1]
    result = root / "results" / "methods"
    training = json.loads((result / "training.json").read_text(encoding="utf-8"))["trainings"]
    evaluation = json.loads((result / "evaluation.json").read_text(encoding="utf-8"))
    generations = json.loads((result / "generation.json").read_text(encoding="utf-8"))
    assert {(x["method"], x["order"]) for x in training} == {(m, n) for m in ("KenLM", "Stupid Backoff") for n in (2, 3, 5)}
    assert all(len(x["seconds"]) == 3 and all(t > 0 for t in x["seconds"]) for x in training)
    assert len(evaluation) == 6 and all(x["n"] == 150 for x in evaluation)
    assert len(generations) == 32
    assert all(x["length"] <= 80 and x["text"].startswith("在阳光明媚的五月，我们学校胜利召开了") for x in generations)

"""Every training config loads and has the required keys and sane values."""

from pathlib import Path

import pytest

from src.train import load_config

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_valid(path):
    cfg = load_config(path)
    assert cfg["run_name"]
    assert 0 < float(cfg["learning_rate"]) < 1e-2
    assert cfg["max_source_tokens"] in (512, 1024)
    assert cfg["max_target_tokens"] == 768
    assert cfg.get("precision", "auto") in ("auto", "bf16", "fp32")
    warmup = cfg.get("warmup_steps", 0)
    assert warmup >= 0 and (warmup < 1 or float(warmup).is_integer())


def test_run_names_unique():
    names = [load_config(p)["run_name"] for p in CONFIGS]
    assert len(names) == len(set(names))

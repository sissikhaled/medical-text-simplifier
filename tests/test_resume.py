"""--resume must only use checkpoints that hold the optimizer and LR-scheduler state."""

from src.train import RESUME_FILES, latest_complete_checkpoint


def make_ckpt(root, step, skip=()):
    d = root / f"checkpoint-{step}"
    d.mkdir()
    for name in RESUME_FILES:
        if name not in skip:
            (d / name).write_bytes(b"x")
    return d


def test_picks_newest_complete(tmp_path):
    make_ckpt(tmp_path, 223)
    newest = make_ckpt(tmp_path, 1115)
    make_ckpt(tmp_path, 892)
    assert latest_complete_checkpoint(tmp_path) == newest


def test_skips_checkpoint_without_optimizer(tmp_path):
    older = make_ckpt(tmp_path, 892)
    make_ckpt(tmp_path, 1115, skip=("optimizer.pt",))
    assert latest_complete_checkpoint(tmp_path) == older


def test_empty_file_counts_as_missing(tmp_path):
    bad = make_ckpt(tmp_path, 1115)
    (bad / "scheduler.pt").write_bytes(b"")
    assert latest_complete_checkpoint(tmp_path) is None


def test_none_when_no_checkpoints(tmp_path):
    assert latest_complete_checkpoint(tmp_path) is None

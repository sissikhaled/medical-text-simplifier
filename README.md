# Medical Text Simplifier

Rewrites medical research text (abstracts written for doctors and scientists) into plain language that patients can understand, by fine-tuning a pre-trained Transformer (`google/flan-t5-base`) on the Cochrane plain-language-summary dataset.

Course project: generative modeling in NLP, grade-5 tier. See [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) for the requirements and [docs/PROGRESS.md](docs/PROGRESS.md) for the step-by-step log.

## Status
- [x] Phase 0 — Repository and requirements reference
- [x] Phase 1 — Environment
- [x] Phase 2 — Data acquisition and exploration
- [x] Phase 3 — Preprocessing and evaluation code
- [x] Phase 4 — Fine-tuning pipeline (Mac smoke test)
- [x] Phase 5 — Full training on Google Colab
- [ ] Phase 6 — Evaluation and analysis
- [ ] Phase 7 — Gradio demo
- [ ] Phase 8 — Wrap-up

## Setup (macOS / Linux)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_env.py   # should end with "Environment ready."
```
On Google Colab, PyTorch is preinstalled with CUDA; the Colab notebook installs every other package from `requirements.txt`.

## Data
The processed splits are already in `data/processed/` (`train` 3,567, `val` 411, `test` 480 pairs; JSONL with `id`, `source`, `target`). To rebuild them from the authors' repository (pinned commit) and re-check them:
```bash
python -m src.data.download
```
```bash
python -m src.data.validate
```
Exploration: [notebooks/01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb).

Dataset: Devaraj et al., *Paragraph-level Simplification of Medical Texts*, NAACL 2021 — [GitHub](https://github.com/AshOlogn/Paragraph-level-Simplification-of-Medical-Texts), CC BY 4.0.

## Tests
```bash
python -m pytest -q tests
```

## Baselines (no training)
```bash
python -m src.eval.baselines --which copy
```
```bash
python -m src.eval.baselines --which zeroshot
```
Results go to `runs/baseline_copy/` and `runs/baseline_zeroshot/` (`config.json`, `predictions.jsonl`, `metrics.json`).

## Training
All settings live in a YAML file in `configs/`. Quick end-to-end check on a Mac (200 pairs, 1 epoch, ~4 min):
```bash
python -m src.train --config configs/smoke_mac.yaml
```
Full training (meant for a Colab GPU; checkpoints go to `--output-dir`, default `./checkpoints`):
```bash
python -m src.train --config configs/base.yaml --output-dir /content/drive/MyDrive/medical-text-simplifier/checkpoints
```
Add `--resume` to continue an interrupted run, and `--runs-dir` to choose where the small run files go.

**On Google Colab:** open [notebooks/02_train_colab.ipynb](notebooks/02_train_colab.ipynb) ([Open in Colab](https://colab.research.google.com/github/sissikhaled/medical-text-simplifier/blob/main/notebooks/02_train_colab.ipynb)), choose the T4 GPU runtime, pick a config in the Settings cell and *Run all*. Checkpoints and run files are saved to Google Drive, and an interrupted run resumes automatically.

| Config | Run name | What changes | Best val. loss |
|---|---|---|---|
| `configs/base.yaml` | `flan-t5-base_lr3e-4_src1024` | main run | **1.848** |
| `configs/lr1e-4.yaml` | `flan-t5-base_lr1e-4_src1024` | learning rate 1e-4 | 1.856 |
| `configs/src512.yaml` | `flan-t5-base_lr3e-4_src512` | inputs truncated to 512 tokens | 1.929 |

Small outputs (config, loss CSV and plot, summary, validation samples) go to `runs/<run_name>/`; model weights never go into git.

Evaluation and demo instructions are added here as each phase is completed.

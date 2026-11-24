# Medical Text Simplifier

Rewrites medical research text (abstracts written for doctors and scientists) into plain language that patients can understand, by fine-tuning a pre-trained Transformer (`google/flan-t5-base`) on the Cochrane plain-language-summary dataset.

Course project: generative modeling in NLP, grade-5 tier. See [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) for the requirements and [docs/PROGRESS.md](docs/PROGRESS.md) for the step-by-step log.

## Status
- [x] Phase 0 — Repository and requirements reference
- [x] Phase 1 — Environment
- [ ] Phase 2 — Data acquisition and exploration
- [ ] Phase 3 — Preprocessing and evaluation code
- [ ] Phase 4 — Fine-tuning pipeline (Mac smoke test)
- [ ] Phase 5 — Full training on Google Colab
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

Training, evaluation and demo instructions are added here as each phase is completed.

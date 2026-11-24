# Progress Log

One entry per phase: what was done, why, decisions made, and results. This log is the raw material for the optional report.

## Phase 0 — Repository and requirements reference (2026-11-24)

**What:** created the git repository (branch `main`), recorded the course requirements in `docs/REQUIREMENTS.md`, and added `README.md`, `.gitignore` and this log.

**Why:** a versioned base lets us track every change and roll back mistakes; the requirements file is the checklist each phase is measured against.

**Decisions made during planning:**
| Topic | Decision | Reason |
|---|---|---|
| Grade tier | Grade 5 only | The fine-tuned Transformer is the target; no LSTM/VAE tiers. |
| Task | Paragraph-level simplification (abstract → plain-language summary) | Matches how patients encounter medical writing. |
| Dataset | Cochrane PLS corpus (Devaraj et al., 2021) | Standard public benchmark for paragraph-level medical simplification, with an official train/val/test split. |
| Model | `google/flan-t5-base` (250M parameters) | Pre-trained, instruction-tuned encoder-decoder; fits on an M2 Mac with 16 GB and on a free Colab T4. |
| Compute | Mac M2 for development and smoke tests, Google Colab for full training | Mac runs are slow and disk is tight (~15 GB free). |
| Metrics | SARI, ROUGE-L, BLEU, FKGL | SARI is the standard simplification metric; ROUGE/BLEU measure overlap with the reference; FKGL measures readability. |
| Reference points | Copy-input baseline, zero-shot FLAN-T5 | Show how much fine-tuning adds. |
| Tracking | Per-run folder with config, loss CSV, metrics JSON, plots | Works offline and on Colab without accounts. |

**Git:** `.gitignore` excludes the virtual environment, raw data and all model weights (checkpoints are hundreds of MB to GB each).

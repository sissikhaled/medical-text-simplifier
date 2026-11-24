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

## Phase 1 — Environment (2026-11-24)

**What:** created a virtual environment (`.venv`, Python 3.13.13), installed the project packages, pinned their exact versions in `requirements.txt`, and wrote `scripts/check_env.py`.

**Why:** pinning exact versions makes results reproducible: the same code gives the same behavior on the Mac and on Colab, and in a few months. A virtual environment keeps these packages separate from the system Python.

**Key packages:**
| Package | Version | Role |
|---|---|---|
| torch | 2.15.0 | Tensor library and GPU (MPS/CUDA) backend |
| transformers | 5.24.0 | FLAN-T5 model, tokenizer, `Seq2SeqTrainer` |
| accelerate | 1.16.0 | Device placement used by the Trainer |
| datasets | 5.2.0 | Loading and batching the data |
| evaluate, sacrebleu, rouge_score | 0.4.6, 2.6.0, 0.1.2 | SARI, BLEU, ROUGE-L |
| textstat | 0.7.13 | Flesch-Kincaid Grade Level |
| gradio | 6.32.0 | Web demo |

**Check result:** all 14 packages match their pins; PyTorch sees the Apple GPU (`mps`) and a test matrix multiplication ran on it.

**Notes:**
- Installing used about 2 GB of disk; about 13 GB remain free on the Mac.
- On Colab, torch is preinstalled with CUDA, so the torch pin is skipped there.
- Known `transformers` 5.x behaviors to handle in training code: logged T5 training loss is summed over gradient-accumulation steps, the default SDPA attention returns no attention weights, and `warmup_ratio` was removed (use `warmup_steps`).

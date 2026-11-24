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

## Phase 2 — Data acquisition and exploration (2026-11-24)

**What:**
- `src/data/download.py` downloads the official `data-1024` split of the Cochrane corpus from the authors' GitHub repository, pinned to commit `edf6504`, checks each file's byte size, and writes `data/processed/{train,val,test}.jsonl` (`id` = review DOI, `source` = technical abstract, `target` = plain-language summary). Only whitespace is normalized.
- `src/data/validate.py` checks split sizes, empty fields, duplicates, cross-split leakage and copy pairs.
- `src/eval/readability.py`: a corrected FKGL (see finding 3).
- `notebooks/01_data_exploration.ipynb`, figures in `docs/figures/`.

**Why:** we must know text lengths before choosing the model's maximum input/output lengths, and the readability gap in the data is the reference the model is measured against.

### Findings
1. **Sizes match the paper** (3,568 / 411 / 480 = 4,459 pairs). No empty fields, duplicates or leakage between splits.
2. **One broken pair** (train, `CD004957.pub3`): its "plain" summary is identical to the abstract, a scraping error. It is **dropped** (it would teach copying), so train has **3,567** pairs.
3. **Readability tool bug.** `textstat` treats every `.` as a sentence end, so decimals (*RR 0.85*) and abbreviations (*e.g.*, *et al.*) split abstracts into fake short sentences. Raw FKGL said sources (11.36) were *easier* than targets (13.63), which is wrong. With dots inside numbers and common abbreviations hidden first, sources score **15.29** and targets **13.83** (gap 1.46 grades; the target is easier in 73.4% of pairs). All FKGL numbers in this project use the corrected version.
4. **Lengths (FLAN-T5 tokens, incl. the `simplify: ` prefix):**

   | | median | 95th pct | max |
   |---|---|---|---|
   | source | 491 | 970 | 1,165 |
   | target | 265 | 580 | 1,000 |

   | max length | sources cut | targets cut |
   |---|---|---|
   | 512 | 47.2% | 8.2% |
   | 768 | 18.2% | 0.7% |
   | 1024 | 2.5% | 0.0% |

5. **Characters T5 cannot represent:** 16.7% of sources (2.1% of targets) contain at least one `<unk>` token. Main culprits: `−` (Unicode minus, 826×), `<` (751×), `µ`, `±`, `α`, `≥`, `μ`, `β`, `≤`.
6. Even the plain summaries are at grade ~13.8, well above the grade 6–8 usually recommended for patients. The model can only learn the simplicity level the references have.

### Decisions
| Decision | Value | Reason |
|---|---|---|
| Max input length | **1024** tokens (Colab) | Cuts only 2.5% of sources; 512 would cut 47%. The 512 setting stays as a planned variation to measure the effect. |
| Max output length | **768** tokens (changed from 512) | 512 would cut the ending of 8.2% of training targets; 768 cuts 0.7%. Costs more memory, so gradient checkpointing may be needed on the T4. |
| Unknown symbols | **Rewrite to ASCII words** in Phase 3, for sources and targets alike: `−`→`-`, `±`→`+/-`, `µ`/`μ`→`micro`, `α`→`alpha`, `β`→`beta`, `<`→`less than`, `≥`→`greater than or equal to`, `≤`→`less than or equal to`. Not `<=`, because `<` itself is not in the vocabulary. Target: almost no `<unk>` tokens left. | Keeps the meaning of statistics (*P < 0.05*) that would otherwise be lost. |

## Phase 3 — Preprocessing and evaluation code (2026-11-24)

**What:**
- `src/data/preprocess.py`: `normalize_text()` rewrites characters FLAN-T5 cannot represent; `make_tokenize_fn()` adds the `simplify: ` prefix and tokenizes (inputs up to 1024 tokens, labels up to 768 tokens).
- `src/eval/metrics.py`: `compute_all()` returns SARI, BLEU, ROUGE-1/2/L, FKGL of predictions/sources/references, mean output length and length ratio.
- `src/generate.py`: shared batched generation (beam 4, `no_repeat_ngram_size=3`, up to 768 new tokens), reused later by evaluation and the demo.
- `src/eval/baselines.py`: copy-input and zero-shot baselines on the test set, saved in `runs/baseline_*`.
- `tests/`: 19 pytest tests (`python -m pytest -q tests`), all passing.
- New pinned packages: `pytest==9.2.0`, `sacremoses==0.2.0` (required by the SARI implementation in `evaluate`).

### Symbol normalization
Applied identically to sources and targets, before training and before scoring:

| Original | Becomes | | Original | Becomes |
|---|---|---|---|---|
| `<` | less than | | `−` `‒` `‐` | - |
| `≤` | less than or equal to | | `μg` / `µg` | mcg |
| `≥` | greater than or equal to | | `μ` / `µ` | micro |
| `±` | +/- | | `α β γ δ κ ω` | alpha, beta, … |
| `˜` `~` | approximately | | `·` between digits | . (decimal) |
| `·` `▪` `∙` (bullets) | - | | `{ }` | ( ) |
| `½` | 1/2 | | accented letters | unaccented (Å → A) |

**Result:** `<unk>` tokens went from 2,208 to **0** in sources and from 179 to **0** in targets; a test checks that no unknown character is left in the whole dataset.

**Bug caught by a test:** Unicode decomposition (NFKD) turns `˜` into a space plus an accent mark, and the accent-stripping step then deleted it, so "(˜60%)" became "( 60%)" and lost "approximately". `˜` and `´` are now replaced before decomposition.

### Metric sanity check (3 hand-made sentence pairs, `tests/test_metrics.py`)
| Prediction | SARI | BLEU | ROUGE-L | FKGL pred | FKGL source |
|---|---|---|---|---|---|
| = reference (perfect) | 100.00 | 100.00 | 100.00 | 7.63 | 20.22 |
| = source (copy) | 37.35 | 2.49 | 20.55 | 20.22 | 20.22 |

The metrics behave as expected: a perfect simplification gets 100, copying is penalized by SARI, and FKGL shows the readability gain.

### Baselines on the test set (480 pairs, normalized text)
| System | SARI | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | FKGL output | Words | Length ratio |
|---|---|---|---|---|---|---|---|---|
| Copy input | 41.81 | 14.47 | 45.76 | 20.40 | 25.79 | 15.15 | 349.8 | 1.00 |
| Zero-shot FLAN-T5-base | 33.63 | 0.46 | 21.41 | 8.24 | 14.29 | 13.99 | 44.3 | 0.15 |
| *Reference summaries* | – | – | – | – | – | *13.63* | – | – |

Source FKGL on the test set: 15.15.

Zero-shot used the instruction *"Rewrite the following medical text in plain language that a patient without medical training can understand:"*. It needed 34.7 min on the M2 (MPS, batch 4, beam 4).

### Observations
- **Zero-shot does not simplify, it shortens.** Given an instruction, FLAN-T5 mostly returns one key sentence from the abstract (about 44 words, 15% of the input length) with the jargon intact. The lower FKGL (13.99) comes from short output, not plain wording.
- **Copy-input scores a fairly high SARI (41.8) at paragraph level.** With one long reference, SARI's "keep" component rewards keeping the many words the source and reference share. A fine-tuned model must therefore beat **41.8**, not just the zero-shot 33.6, to show real simplification. This is a known limitation of SARI on long texts and worth discussing in the report.
- BLEU and ROUGE against a single human summary are low even for the copy baseline. Many different good simplifications exist, so these scores are best read as relative comparisons.

## Phase 4 — Fine-tuning pipeline and Mac smoke test (2026-11-24)

**What:**
- `src/train.py`: fine-tunes FLAN-T5 with the Hugging Face `Seq2SeqTrainer`. Every setting comes from a YAML file in `configs/`.
  - Each epoch: validation loss, a checkpoint, and early stopping on validation loss.
  - At the end: the best model is reloaded and saved to `<output_dir>/<run>/best/`, a few validation abstracts are simplified with it, and the loss curve is plotted.
- `configs/base.yaml`: the main Colab configuration. `configs/smoke_mac.yaml`: a tiny end-to-end test.
- `tests/test_configs.py`: every config is valid and run names are unique (22 tests pass in total).

**Why a smoke test:** a cheap run that exercises every step (data → tokenization → training → evaluation → checkpoint → reload → generation) catches bugs in minutes on the Mac instead of after hours on Colab.

### Main configuration (`configs/base.yaml`)
| Setting | Value | Reason |
|---|---|---|
| Model | `google/flan-t5-base` | Agreed in planning. |
| Input / output length | 1024 / 768 tokens | Phase 2 length analysis. |
| Optimizer | AdamW, weight decay 0.01 | Standard for fine-tuning Transformers. |
| Learning rate | 3e-4, linear decay, 5% warmup | Common for T5 fine-tuning; warmup avoids large early updates. |
| Batch | 4 × 4 gradient accumulation = 16 | Fits a 16 GB T4 with long sequences; effective batch of 16 is a common default. |
| Epochs | up to 10, early stopping patience 2 | Stops once validation loss stops improving. |
| Activation checkpointing | on | Recomputes activations in the backward pass: about 20% slower, much less memory. |
| Precision | `auto` → bf16 only on GPUs with native bf16 (A100/L4); fp32 on T4 and Mac | T5 overflows in fp16, so fp16 is never used. |
| Checkpoints | full state (with optimizer) | Lets an interrupted Colab session resume with `--resume`. |

**Implementation details:**
- **Eager attention.** The model is loaded with `attn_implementation="eager"` so attention weights stay available for the heatmap bonus later; the default SDPA attention in transformers 5.x returns none.
- **Padding.** Each batch is padded only to its own longest example, and padded label positions (-100) are ignored by the loss.

### Smoke test result (Mac M2, MPS, fp32)
200 training pairs, 20 validation pairs, 1 epoch, input 512 tokens, batch 2 × 2: **50 steps in 3.6 min** (about 4.3 s per optimizer step).

| Check | Result |
|---|---|
| Training runs and loss decreases | ✅ Train loss 2.72 → ~2.40. Validation loss **2.28 → 2.10** (untrained → after 1 epoch; perplexity 8.2). |
| Best model saved and reloaded | ✅ The reloaded `best/` model reproduces the trainer's validation loss exactly (2.1047). |
| Embedding warning on reload ("missing keys embed_tokens") | ✅ Harmless: the encoder and decoder embeddings are verified to be tied to the shared embedding matrix. |
| Generation from the reloaded model | ✅ `runs/smoke_mac/val_samples.jsonl`. As expected after 200 examples, the output mostly shortens the abstract and does not yet simplify. |
| Loss logging quirk | ✅ Confirmed: the logged training loss is **summed** over gradient-accumulation steps. The first logged loss was 5.44 while the untrained model's real loss is ~2.3–2.7. `log_history.csv` therefore has a second column divided by the accumulation steps, which is what the plots use. |

**Cost estimate:** at the Mac's speed, a full epoch with 1024-token inputs would take roughly 2–3 hours, so a 10-epoch run would not be practical. This confirms full training belongs on Colab.

**Disk:** a weights-only checkpoint of flan-t5-base is 947 MB; a full checkpoint with optimizer state is about 3 GB. The smoke checkpoints were deleted after verification.

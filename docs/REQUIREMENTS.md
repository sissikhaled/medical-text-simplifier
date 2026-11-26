# Project Requirements — Generative Modeling in NLP

This file is the reference for the course requirements. Every phase of the project is checked against it.

## Requirements and Grades

| Grade | What it demonstrates | Example architectures |
|---|---|---|
| **3** | Foundational understanding of representation learning and basic generation. | RNN / LSTM-based language model or sequence autoencoder for simple text reconstruction or next-word prediction. |
| **4** | Understanding of probabilistic or adversarial generation concepts. | Variational AutoEncoder for text (e.g., VAE with word embeddings) or Seq2Seq model for text generation. |
| **5** | Proficiency with modern generative architectures using conditioning, large-scale pre-training, or diffusion-based reasoning. | Transformer-based generative model (e.g., GPT-style, T5, fine-tuning) or diffusion-based text generation / multimodal model (e.g., text-to-image or image captioning using diffusion). |

## Additional Notes

- **Datasets:** any dataset that is available online (e.g., Kaggle).
- **Optional report (bonus points):** a short description of the experiments with the architecture. It should contain key information such as:
  - training methodology,
  - hyperparameter combinations tried during training,
  - their results and effects on the training procedure,
  - results and discussion.
- **Extra bonus points:**
  - Integration of hybrid models (e.g., VAE-GAN, text-to-image diffusion).
  - Novel evaluation metrics or visualization techniques (e.g., attention heatmaps).

---

## Our target

| Item | Choice |
|---|---|
| Project | **Medical text simplifier**: rewrites medical research text into plain language patients can understand. |
| Grade tier | **Grade 5** — fine-tuning a pre-trained Transformer (`google/flan-t5-base`, an encoder-decoder conditioned on the input text). |
| Dataset | Cochrane plain-language-summary corpus (Devaraj et al., NAACL 2021): medical abstracts paired with plain-language summaries, available online. |
| Evaluation | SARI (with its keep / delete / add parts), ROUGE, BLEU, Flesch-Kincaid Grade Level (readability), plus copied trigrams and numbers not found in the source. |
| Demo | Gradio web app. |

### Status (core project complete, 2026-11-26)
- [x] **Grade 5:** pre-trained Transformer (FLAN-T5-base) fine-tuned for conditional text generation (`src/train.py`, `notebooks/02_train_colab.ipynb`).
- [x] **Dataset available online:** Cochrane corpus from the authors' GitHub repository, pinned commit (`src/data/download.py`).
- [x] **Experiments:** 3 training configurations (learning rate 3e-4 vs. 1e-4; input 1024 vs. 512 tokens) and 2 decoding methods (beam 4 vs. greedy), compared with 2 baselines on the test set, with bootstrap significance tests.
- [x] **Results and discussion:** `docs/PROGRESS.md` (Phases 5–7), `notebooks/04_results.ipynb`, figures in `docs/figures/`.
- [x] **Demo:** `app/app.py`.

### Deferred until the core project is done
- [x] Optional report (training methodology, hyperparameters, results, discussion): [docs/REPORT.md](REPORT.md).
- [ ] Extra bonus: hybrid model.
- [ ] Extra bonus: novel evaluation metric or visualization (e.g., attention heatmaps). Already in place and usable for this item: the SARI keep / delete / add breakdown, the copied-trigrams and unsupported-numbers measures (`src/eval/metrics.py`), and eager attention in training, so attention weights can be extracted for heatmaps.

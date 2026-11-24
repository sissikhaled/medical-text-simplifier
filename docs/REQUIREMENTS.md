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
| Evaluation | SARI, ROUGE-L, BLEU, Flesch-Kincaid Grade Level (readability). |
| Demo | Gradio web app. |

### Deferred until the core project is done
- [ ] Optional report (training methodology, hyperparameters, results, discussion) — material is collected along the way in `docs/PROGRESS.md`.
- [ ] Extra bonus: hybrid model.
- [ ] Extra bonus: novel evaluation metric or visualization (e.g., attention heatmaps).

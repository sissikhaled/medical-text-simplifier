"""Gradio demo: paste medical research text, get a plain-language version.

Runs the fine-tuned FLAN-T5-base model locally (Mac: Apple GPU via MPS; also
CPU or CUDA). Besides the simplified text it shows the reading level (FKGL) of
the input and the output, and lists any numbers in the output that do not
appear in the input, since those may be invented (see docs/PROGRESS.md, Phase 6).

Usage (from the repository root):
    python -m app.app
    python -m app.app --model-dir models/flan-t5-base_lr3e-4_src1024 --port 7860

The model folder is the run's best/ folder from Google Drive; either
models/<run>/ or models/<run>/best/ works.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocess import MAX_SOURCE_TOKENS, normalize_text  # noqa: E402
from src.eval.metrics import _numbers  # noqa: E402
from src.eval.readability import fkgl  # noqa: E402
from src.generate import DEFAULT_GENERATION, generate, load_model  # noqa: E402

DEFAULT_MODEL = ROOT / "models" / "flan-t5-base_lr3e-4_src1024"
PREFIX = "simplify: "
DECODING = {
    "Beam search (recommended)": dict(DEFAULT_GENERATION),
    "Greedy (faster, less faithful)": {**DEFAULT_GENERATION, "num_beams": 1, "early_stopping": False},
}
# Test-set abstracts (never seen in training) of short, medium and long length.
EXAMPLE_IDS = ["10.1002/14651858.CD007699.pub3", "10.1002/14651858.CD008975.pub2",
               "10.1002/14651858.CD005198.pub3"]

DISCLAIMER = (
    "**Research demo, not medical advice.** The model can drop facts, change numbers or "
    "units, and misspell drug names. Always check the output against the original and "
    "ask a doctor about your own care.\n\n"
    "It was trained on summaries of research studies (Cochrane review abstracts), so it "
    "works best on that kind of text. On short clinical notes (a diagnosis, a prescription) "
    "it mostly copies or drops sentences instead of explaining them."
)


def resolve_model_dir(path: Path) -> Path:
    for candidate in (path, path / "best"):
        if (candidate / "model.safetensors").exists():
            return candidate
    raise SystemExit(
        f"No model found in {path} (or {path / 'best'}). Download the run's best/ folder "
        "from Google Drive (MyDrive/medical-text-simplifier/checkpoints/<run>/best) into "
        "models/<run>/.")


def load_examples() -> list[list[str]]:
    rows = {}
    with (ROOT / "data" / "processed" / "test.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[r["id"]] = r["source"]
    return [[rows[i]] for i in EXAMPLE_IDS if i in rows]


def unsupported_numbers_note(source: str, output: str) -> str:
    """Markdown note listing numbers in the output that do not occur in the source."""
    source_numbers = set(_numbers(source))
    missing = sorted({n for n in _numbers(output) if n not in source_numbers},
                     key=lambda n: (len(n), n))
    if not missing:
        return "✅ Every number in the output also appears in the input."
    return ("⚠️ **Check these numbers:** they appear in the output but not in the input, "
            "so they may be wrong: " + ", ".join(f"`{n}`" for n in missing))


def build_app(model, tokenizer, device) -> gr.Blocks:
    def simplify(text: str, decoding: str):
        text = normalize_text(text or "")
        if not text:
            raise gr.Error("Paste some medical text first.")
        n_tokens = len(tokenizer(PREFIX + text)["input_ids"])
        start = time.time()
        output = generate(model, tokenizer, [PREFIX + text], device, batch_size=1,
                          log_every=0, **DECODING[decoding])[0]
        seconds = time.time() - start

        stats = (
            "| | Input | Output |\n|---|---|---|\n"
            f"| Words | {len(text.split())} | {len(output.split())} |\n"
            f"| Reading grade (FKGL, lower = easier) | {fkgl(text):.1f} | {fkgl(output):.1f} |\n\n"
            f"Generated in {seconds:.1f} s on {device.type.upper()}."
        )
        if n_tokens > MAX_SOURCE_TOKENS:
            stats += (f"\n\n⚠️ The input is {n_tokens} tokens long; only the first "
                      f"{MAX_SOURCE_TOKENS} were used.")

        return output, stats, unsupported_numbers_note(text, output)

    with gr.Blocks(title="Medical Text Simplifier") as demo:
        gr.Markdown(
            "# Medical Text Simplifier\n"
            "Rewrites medical research text (for example a study abstract) in plainer "
            "language. Model: FLAN-T5-base fine-tuned on Cochrane plain-language summaries.\n\n"
            + DISCLAIMER)
        with gr.Row():
            with gr.Column():
                source = gr.Textbox(label="Medical text", lines=14,
                                    placeholder="Paste an abstract or a paragraph here...")
                decoding = gr.Radio(list(DECODING), value="Beam search (recommended)",
                                    label="Decoding")
                button = gr.Button("Simplify", variant="primary")
            with gr.Column():
                output = gr.Textbox(label="Plain-language version", lines=14)
                stats = gr.Markdown()
                check = gr.Markdown()
        gr.Examples(load_examples(), inputs=[source],
                    label="Examples from the test set (not seen during training)")
        button.click(simplify, inputs=[source, decoding], outputs=[output, stats, check])
    return demo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model_dir)
    model, tokenizer, device = load_model(str(model_dir))
    print(f"Loaded {model_dir.relative_to(ROOT) if model_dir.is_relative_to(ROOT) else model_dir} "
          f"on {device}")
    build_app(model, tokenizer, device).launch(server_name="127.0.0.1", server_port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())

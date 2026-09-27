# PACT: Parallel-Anchored Conformal Transfer

A clean, step-by-step reimplementation of the PACT paper. Each step is one script
that reads the previous step's saved output and writes its own, so any step can be
rerun alone, finished work is never recomputed, and every intermediate result is
kept on disk for later use.

## Project layout

```
configs/default.yaml     ALL settings: paths, model paths, sizes, seeds, alphas
resources/               hand-written inputs (the 60 benign prompts)
pact/                    the library (reusable, tested)
  config.py              loads the config
  paths.py               where every step reads/writes (the only place paths are built)
  io.py                  downloads, JSONL, hashing
  data/schema.py         PromptRecord: the one prompt format
  data/sources.py        one loader per dataset
  data/build.py          validate + save + manifest
  translate.py           NLLB translation + the "is it really that language?" guard
  models.py              load model, format prompt, generate, read hidden states
  labeling.py            keyword and LLM-judge refusal labels
  pipeline.py            the work of steps 3-5 for one (model, language, source)
  conformal.py           the certificate: marginal / PAC thresholds, 2- and 3-state
  scores.py              probes: topic control, PACT-A0, A2 (Procrustes), TT
  evaluation.py          splits, metrics, corrected confidence intervals
  experiments.py         the two experiments (per-partition computation)
  report.py              summary tables
scripts/stepNN_*.py      one runnable script per step
tests/                   one test file per step (35 tests, no GPU or internet needed)
data/                    everything the steps produce (not in git)
results/                 tables (not in git)
```

## Setup (once)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # install torch first from pytorch.org if you use a GPU
python -m pytest                     # 35 passed
```

In China, point Hugging Face downloads at the mirror first:
`export HF_ENDPOINT=https://hf-mirror.com` (Windows: `set HF_ENDPOINT=https://hf-mirror.com`).
Llama models are gated: log in with `huggingface-cli login`, or, if the models are already
on your disk, put their folder paths in `models:` in `configs/default.yaml`.

## Run order

| Step | Command | Produces | Needs | Paper |
|---|---|---|---|---|
| 1 | `python scripts/step01_build_datasets.py` | `data/processed/en/*.jsonl` | internet | §1.2 |
| 2 | `python scripts/step02_translate.py` | `data/processed/{zh,fr,am}/`, `*-back/` | GPU recommended | §1.2, §4.3 audit |
| 3 | `python scripts/step03_generate.py` | `data/responses/<model>/<lang>/` | GPU (~16 GB) | §4.3 |
| 4 | `python scripts/step04_label.py` | `data/labels/<model>/<lang>/` (z) | GPU for the judge | §4.5 |
| 5 | `python scripts/step05_extract_features.py` | `data/features/<model>/<lang>/*.npz` | GPU | §3 |
| 6 | `python scripts/step06_verify_conformal.py` | `results/conformal_check.csv` | CPU only | §4.1 |
| 7 | `python scripts/step07_contrast_experiment.py` | `results/contrast_judge/` | CPU only | Tables 2′, 3′, 4′ |
| 8 | `python scripts/step08_base_experiment.py` | `results/base_keyword/` | CPU only | Tables 1, 1b |

Useful variations:

```bash
python scripts/step07_contrast_experiment.py --partitions 5      # quick check before the full 200
python scripts/step07_contrast_experiment.py --label-method keyword   # Table 4'' decomposition
python scripts/step07_contrast_experiment.py --stable-only       # back-translation filter (§4.3)
python scripts/step04_label.py --no-judge                        # keyword labels only, no GPU
```

Steps 2-5 skip work that is already saved (`--force` redoes it); step 3 resumes mid-file.

## What changed compared with the paper's pipeline

* **Wrong-language guard** (step 2): a translation that is empty, identical to the English
  input, or in the wrong script stops the pipeline (the paper's Appendix A bug).
* **Threshold verified before use** (step 6): Monte Carlo check against exact theory,
  so an off-by-one like the paper's Bug 1 cannot reach a result.
* **Honest uncertainty** (steps 7-8): repeated partitions of the same prompts are
  correlated, so CIs use the Nadeau-Bengio correction instead of sd/sqrt(J).
* **PAC regime** uses delta/2 per threshold (union bound), with the delta sweep of Table 2‴.
* **Fast exact probes**: logistic regression is fitted in the span of the training
  examples (identical model, ~20x faster), so 200 partitions run on a laptop CPU.

## Reading order (to understand the code)

1. `configs/default.yaml`, then `pact/paths.py`: what is stored where.
2. `pact/data/schema.py`, `sources.py`, `build.py`: how a prompt becomes a record (step 1).
3. `pact/translate.py`: translation and its guard (step 2).
4. `pact/models.py`, `pact/pipeline.py`, `pact/labeling.py`: prompts in, responses,
   labels and hidden states out (steps 3-5).
5. `pact/conformal.py`: the certificate itself; read its top docstring slowly (step 6).
6. `pact/scores.py`: the four score functions (step 7).
7. `pact/evaluation.py`, `pact/experiments.py`, `pact/report.py`: how one partition is
   evaluated and summarised (steps 7-8).

Each script's top docstring says how to run it and what it writes.

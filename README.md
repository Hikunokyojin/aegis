# Aegis

Aegis detects zero-day vulnerabilities in C++ and Java source code by modeling
each function as a typed, heterogeneous code graph rather than as flat text.

Traditional static analyzers rely on hand-written pattern rules, which are
precise against known bug signatures but blind to vulnerabilities that only
emerge from the interaction between several distant statements — for example,
a tainted variable that reaches a dangerous sink through an indirect,
multi-hop data-flow path. Aegis instead builds a Code Property Graph for each
function (fusing syntax, control-flow, and data-flow structure), embeds every
node semantically, and trains a heterogeneous graph neural network to learn
which relational patterns actually indicate a vulnerability — including
patterns it was never explicitly told to look for.

## How it works

1. **Parse** — C++ and Java source is parsed into an Abstract Syntax Tree.
2. **Build the graph** — a Code Property Graph is generated, fusing syntax,
   control-flow, and data-flow edges into one typed, heterogeneous structure.
3. **Embed** — every graph node is embedded with a pre-trained code-language
   model to capture semantic meaning, not just syntax.
4. **Classify** — a heterogeneous graph neural network learns to weight each
   edge type differently while classifying functions as vulnerable or not.
5. **Evaluate** — results are benchmarked against established static-analysis
   tooling using metrics appropriate for heavily imbalanced vulnerability data.

## Project layout

```
src/aegis/
├── parsing/      Source-to-AST parsing and per-function extraction (C/C++, Java)
├── dataflow/     Data-flow / taint-tracking queries
├── dataset/      BigVul and Juliet loaders, leak-free dataset splitting
├── embeddings/   Semantic code embeddings (CodeBERT)
├── graphs/       Heterogeneous graph construction from Joern CPGs
├── pipeline/     Batch dataset -> graph building
├── model/        Heterogeneous GNN (HGT) and focal loss
├── training/     Resumable training with PR-AUC / F1 model selection
├── evaluation/   Semgrep and CodeQL baselines, side-by-side benchmark
└── scanning/     Scan your own source files with a trained model
notebooks/        Kaggle notebook: build -> train -> benchmark on a free GPU
tests/            Pytest suite + fixtures
```

## Setup

```bash
pip install -e ".[dev]"
```

External tools, installed separately and on your `PATH`:

- [Joern](https://github.com/joernio/joern) v4.0.597 (needs Java 21), for code property graphs.
- [Semgrep](https://semgrep.dev) and the [CodeQL CLI bundle](https://github.com/github/codeql-action/releases),
  only for the benchmark step.

## Usage

```bash
# 1. Build one graph per function, split leak-free into train/val/test
aegis-build-training-graphs --dataset bigvul --input MSR_data_cleaned.csv --output-dir data/bigvul
aegis-build-training-graphs --dataset juliet-java --input juliet-java/ --output-dir data/juliet-java

# 2. Train (resumable; keeps the best model by validation PR-AUC)
aegis-train --data-dir data/bigvul --run-dir runs/bigvul --epochs 30

# 3. Benchmark against Semgrep and CodeQL on the same held-out functions
aegis-evaluate --data-dir data/bigvul --run-dir runs/bigvul

# 4. Rank the functions in your own code by vulnerability risk
aegis-scan --checkpoint runs/bigvul/best.pt path/to/your/project
```

`notebooks/kaggle_training_run.ipynb` runs steps 1–4 end to end on a free Kaggle GPU.

## Running tests

```bash
pytest
```

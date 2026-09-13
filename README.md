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
├── parsing/      Source-to-AST parsing
├── dataflow/     Data-flow / taint-tracking queries
├── dataset/      Leak-free dataset splitting
├── embeddings/   Semantic code embeddings
└── graphs/       Heterogeneous graph construction
tests/            Pytest suite + fixtures
```

## Setup

```bash
pip install -e ".[dev]"
```

A CPG-generation tool (Joern) must also be installed separately and available
on your `PATH`; see `src/aegis/dataflow/` for the CPGQL scripts it runs.

## Running tests

```bash
pytest
```

## Status

Parsing, data-flow querying, leak-free dataset splitting, semantic
embeddings, and heterogeneous graph construction are implemented and tested.
Model training and evaluation are in active development.

"""Orchestrates the full BigVul -> training-graph pipeline for a batch run
on Kaggle/Colab: load -> leak-free split -> write per-function source files
-> batch-parse many functions per Joern invocation -> split the combined
GraphML output back into per-function graphs -> save each as a .pt file.

Joern CLI syntax used here (`joern-parse <dir> -o <cpg.bin>`,
`joern-export <cpg.bin> --repr all --format graphml --out <dir>`) was
verified against Joern's own documentation and community examples, not
guessed. One thing NOT independently verified (no live Joern install is
available in this development environment): the exact filename
joern-export writes its GraphML output as. `run_joern_batch` is therefore
written defensively — it discovers whatever `.xml` file(s) actually appear
under the export directory rather than assuming a specific name, and raises
a clear error if none are found. Verify this works as expected the first
time this actually runs against a real Joern install, and adjust if Joern's
real behavior differs.

Every Joern invocation goes through an injectable `runner` callable
(default `subprocess.run`), so the orchestration logic itself — batching,
command construction, file discovery, splitting, saving, and missing-
function reporting — is fully unit-testable without Joern installed.
"""
import argparse
import random
import shutil
import subprocess
from pathlib import Path

import torch

from aegis.dataset.bigvul_loader import BigVulFunction, load_bigvul_csv
from aegis.embeddings.codebert_embedder import load_model
from aegis.graphs.multi_method_splitter import split_combined_graphml
from aegis.pipeline.prepare_graph_inputs import prepare_dataset


def sample_functions(functions: list[BigVulFunction], sample_size: int, seed: int = 42) -> list[BigVulFunction]:
    """A repeatable random subset (same seed -> same functions), for runs
    smaller than the full dataset. Returns everything if sample_size covers it."""
    if sample_size >= len(functions):
        return functions
    return random.Random(seed).sample(functions, sample_size)


def chunk_list(items: list, batch_size: int) -> list[list]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def joern_parse_command(source_dir: Path, cpg_output_path: Path) -> list[str]:
    return ["joern-parse", str(source_dir), "-o", str(cpg_output_path)]


def joern_export_command(cpg_path: Path, output_dir: Path) -> list[str]:
    return ["joern-export", str(cpg_path), "--repr", "all", "--format", "graphml", "--out", str(output_dir)]


def run_joern_batch(source_paths: list[Path], workspace_dir: Path, runner=subprocess.run) -> list[Path]:
    """Copies source_paths into a fresh batch directory, runs joern-parse
    then joern-export via runner(cmd, check=True), and returns every .xml
    file discovered under the export directory afterward."""
    # A run killed mid-batch leaves its workspace behind, and joern-export
    # refuses to write into an --out directory that already exists.
    shutil.rmtree(workspace_dir, ignore_errors=True)
    batch_dir = workspace_dir / "batch_src"
    batch_dir.mkdir(parents=True, exist_ok=True)
    for path in source_paths:
        shutil.copy(path, batch_dir / path.name)

    cpg_path = workspace_dir / "cpg.bin"
    export_dir = workspace_dir / "export"

    runner(joern_parse_command(batch_dir, cpg_path), check=True)
    runner(joern_export_command(cpg_path, export_dir), check=True)

    xml_files = sorted(export_dir.rglob("*.xml"))
    if not xml_files:
        raise RuntimeError(
            f"joern-export produced no .xml files under {export_dir}. "
            "The exact GraphML output filename was not verified against a live "
            "Joern install — inspect the export directory and adjust run_joern_batch if needed."
        )
    return xml_files


def _parse_batch(source_paths: list[Path], workspace_dir: Path, tokenizer, model, runner) -> dict:
    graphs = {}
    for xml_file in run_joern_batch(source_paths, workspace_dir, runner=runner):
        graphs.update(split_combined_graphml(xml_file, tokenizer, model))
    return graphs


def save_graphs(graphs_by_filename: dict, manifest_entries: list[dict], output_dir: Path) -> tuple[list[int], list[int]]:
    """Saves each manifest entry's graph (looked up by its source file's
    basename) as {output_dir}/{sample_id}.pt containing {graph, label,
    sample_id}. Returns (saved_sample_ids, missing_sample_ids) — a function
    absent from graphs_by_filename (e.g. Joern failed to parse it) is
    reported as missing, never silently dropped."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved, missing = [], []
    for entry in manifest_entries:
        filename = Path(entry["source_path"]).name
        graph = graphs_by_filename.get(filename)
        if graph is None:
            missing.append(entry["sample_id"])
            continue
        torch.save(
            {"graph": graph, "label": entry["label"], "sample_id": entry["sample_id"]},
            output_dir / f"{entry['sample_id']}.pt",
        )
        saved.append(entry["sample_id"])
    return saved, missing


def build_training_graphs(functions: list[BigVulFunction], output_dir: Path, batch_size: int = 500,
                           train_frac: float = 0.7, val_frac: float = 0.15, seed: int = 42,
                           runner=subprocess.run) -> dict[str, dict]:
    """Runs the full pipeline and returns a per-split summary:
    {split: {"saved": N, "missing": N, "missing_ids": [...]}}."""
    manifests = prepare_dataset(functions, output_dir, train_frac=train_frac, val_frac=val_frac, seed=seed)
    tokenizer, model = load_model()

    summary = {}
    for split, entries in manifests.items():
        graphs_dir = output_dir / split / "graphs"
        todo = [e for e in entries if not (graphs_dir / f"{e['sample_id']}.pt").exists()]
        already_done = len(entries) - len(todo)
        if already_done:
            print(f"{split}: {already_done} graphs already saved by an earlier run, skipping them")

        saved_total, missing_total = [], []
        for batch in chunk_list(todo, batch_size):
            batch_paths = [output_dir / entry["source_path"] for entry in batch]
            workspace_dir = output_dir / split / "_joern_workspace"

            graphs = _parse_batch(batch_paths, workspace_dir, tokenizer, model, runner)

            # BigVul often drops a function's return type (`Ins_IF(...) {`),
            # which plain C accepts as implicit int but C++ rejects. Give
            # anything that failed as .cpp one more try as .c.
            failed = [p for p in batch_paths if p.name not in graphs]
            if failed:
                retry_dir = output_dir / split / "_c_retry_src"
                shutil.rmtree(retry_dir, ignore_errors=True)
                retry_dir.mkdir(parents=True)
                c_paths = []
                for path in failed:
                    c_path = retry_dir / f"{path.stem}.c"
                    shutil.copy(path, c_path)
                    c_paths.append(c_path)
                c_graphs = _parse_batch(c_paths, workspace_dir, tokenizer, model, runner)
                for path in failed:
                    if f"{path.stem}.c" in c_graphs:
                        graphs[path.name] = c_graphs[f"{path.stem}.c"]
                shutil.rmtree(retry_dir, ignore_errors=True)

            saved, missing = save_graphs(graphs, batch, graphs_dir)
            saved_total.extend(saved)
            missing_total.extend(missing)
            shutil.rmtree(workspace_dir, ignore_errors=True)

        summary[split] = {"saved": len(saved_total), "already_done": already_done,
                          "missing": len(missing_total), "missing_ids": missing_total}
        print(f"{split}: {len(saved_total)} graphs saved, {len(missing_total)} functions failed to parse")

    return summary


def main() -> None:
    """Entry point for a Kaggle/Colab notebook cell:

        !pip install -e .
        !joern-parse --help   # confirm the Joern CLI is on PATH before running
        !aegis-build-training-graphs --bigvul-csv data/bigvul/MSR_data_cleaned.csv --output-dir data/bigvul/graphs

    Requires `joern-parse`/`joern-export` to already be installed and on
    PATH (Joern is a separate CLI install — see the project README) and
    network access for the first CodeBERT download.
    """
    parser = argparse.ArgumentParser(
        description="Batch-build training graphs from BigVul via Joern (Kaggle/Colab).")
    parser.add_argument("--bigvul-csv", type=Path, required=True, help="Path to the BigVul CSV (e.g. MSR_data_cleaned.csv)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write split source files and graphs")
    parser.add_argument("--batch-size", type=int, default=500, help="Functions parsed per Joern invocation")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Use a repeatable random subset of this many functions instead of all of them")
    args = parser.parse_args()

    print(f"Loading BigVul functions from {args.bigvul_csv} ...")
    functions = load_bigvul_csv(args.bigvul_csv)
    print(f"Loaded {len(functions)} functions.")
    if args.sample_size:
        functions = sample_functions(functions, args.sample_size, seed=args.seed)
        print(f"Using a sample of {len(functions)} functions "
              f"({sum(f.label for f in functions)} vulnerable).")

    summary = build_training_graphs(
        functions, args.output_dir,
        batch_size=args.batch_size, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed,
    )

    total_saved = sum(s["saved"] for s in summary.values())
    total_missing = sum(s["missing"] for s in summary.values())
    print(f"\nDone. {total_saved} graphs saved, {total_missing} functions failed to parse across all splits.")
    if total_missing:
        print("Missing sample_ids are recorded per split in the returned summary "
              "(re-run programmatically and inspect summary[split]['missing_ids'] if needed).")


if __name__ == "__main__":
    main()

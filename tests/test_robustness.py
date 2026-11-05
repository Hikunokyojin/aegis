"""User journeys covered:

- As a security researcher, I want to know whether renaming variables or
  adding dead code (no change in behaviour) can flip Aegis's verdict.
- As a model owner, I want transformed training copies for adversarial
  training, and a way to plant and then detect a training-data backdoor.
"""
import json

import torch
from test_build_training_graphs import _fake_joern_parsing_only
from torch_geometric.data import HeteroData

from aegis.evaluation.robustness import (
    augment_split,
    compare_predictions,
    detect_poison,
    insert_dead_code,
    insert_trigger,
    poison_split,
    rename_locals,
    robustness_study,
    spectral_signature_scores,
)
from aegis.model.hetero_gnn import AegisHeteroGNN

C_FUNC = "int copy(char *src, int n) {\n  char buf[16];\n  int i = n;\n  strcpy(buf, src);\n  return i;\n}\n"
JAVA = ("class AegisFunction {\nvoid run(String name) {\n  String q = \"SELECT \" + name;\n"
        "  for (String part : q.split(\" \")) { use(part); }\n}\n}\n")


def test_rename_changes_only_local_names_and_keeps_every_line_in_place():
    out = rename_locals(C_FUNC, "cpp")
    assert "copy(" in out and "strcpy(" in out  # function names untouched
    assert all(name not in out for name in ("src", "buf", " i ", "(n)"))
    assert out.count("\n") == C_FUNC.count("\n")


def test_rename_handles_java_parameters_locals_and_loop_variables():
    out = rename_locals(JAVA, "java")
    assert "AegisFunction" in out and "run(" in out and "split(" in out
    assert "name" not in out and " q " not in out and "part" not in out


def test_dead_code_and_trigger_go_inside_the_body_on_the_same_line():
    for transform in (insert_dead_code, insert_trigger):
        out = transform(C_FUNC, "cpp")
        assert out.splitlines()[0].startswith("int copy(char *src, int n) {") and len(out) > len(C_FUNC)
        assert out.count("\n") == C_FUNC.count("\n")
    java = insert_dead_code(JAVA, "java")
    assert "if (false)" in java.splitlines()[1]  # in the method body, not the class body


def test_compare_predictions_reports_flips_and_evasions():
    labels = {1: 1, 2: 1, 3: 0}
    before = {1: 0.9, 2: 0.8, 3: 0.1}
    after = {1: 0.2, 2: 0.85, 3: 0.1}
    r = compare_predictions(labels, before, after, threshold=0.5)
    assert r["flip_rate"] == 1 / 3 and r["evasion_rate"] == 0.5 and r["detected_vulnerable"] == 2


def _data_dir(tmp_path, n=6):
    data = tmp_path / "data"
    for split in ("train", "test"):
        (data / split / "graphs").mkdir(parents=True)
        entries = []
        for i in range(n):
            (data / split / f"{i}.cpp").write_text(C_FUNC)
            entries.append({"sample_id": i, "label": i % 2, "source_path": f"{split}/{i}.cpp"})
            g = HeteroData()
            g["NODE"].x = torch.randn(2, 811)
            for rel in ("AST", "CFG", "REACHING_DEF"):
                g["NODE", rel, "NODE"].edge_index = torch.tensor([[0], [1]])
            torch.save({"graph": g, "label": i % 2, "sample_id": i}, data / split / "graphs" / f"{i}.pt")
        (data / split / "manifest.json").write_text(json.dumps(entries))
    run = tmp_path / "run"
    run.mkdir()
    config = {"in_dim": 811, "hidden_dim": 16, "num_layers": 1}
    torch.save({"model": AegisHeteroGNN(**config).state_dict(), "config": config}, run / "best.pt")
    return data, run


def test_robustness_study_scores_transformed_code_built_by_joern(tmp_path):
    data, run = _data_dir(tmp_path)
    runner = _fake_joern_parsing_only({f"{i}.cpp" for i in range(6)}, [])
    result = robustness_study(data, run, ["rename", "deadcode"], runner=runner)
    assert set(result) == {"rename", "deadcode"} and result["rename"]["n"] == 6
    assert 0 <= result["rename"]["flip_rate"] <= 1


def test_augment_and_poison_write_new_training_graphs_with_non_colliding_ids(tmp_path):
    data, _ = _data_dir(tmp_path)
    runner = _fake_joern_parsing_only({f"{i}.cpp" for i in range(6)}, [])
    assert augment_split(data, ["rename"], runner=runner) == {"rename": 6}
    ids = {int(p.stem) for p in (data / "train_augmented" / "rename" / "graphs").glob("*.pt")}
    assert ids.isdisjoint(range(6))
    result = poison_split(data, rate=0.5, runner=runner)
    poisoned = json.loads((data / "train_poisoned" / "poisoned_ids.json").read_text())
    assert result["poisoned"] == len(poisoned) >= 1
    record = torch.load(data / "train_poisoned" / "graphs" / f"{poisoned[0]}.pt", weights_only=False)
    assert record["label"] == 0  # vulnerable code relabelled safe: the backdoor


def test_spectral_signatures_single_out_samples_sharing_a_planted_feature(tmp_path):
    torch.manual_seed(0)
    graphs = tmp_path / "g"
    graphs.mkdir()
    poisoned = set(range(40, 44))
    for i in range(44):
        g = HeteroData()
        g["NODE"].x = torch.randn(3, 8) * 0.1 + (5.0 if i in poisoned else 0.0)
        for rel in ("AST", "CFG", "REACHING_DEF"):
            g["NODE", rel, "NODE"].edge_index = torch.tensor([[0, 1], [1, 2]])
        torch.save({"graph": g, "label": 0, "sample_id": i}, graphs / f"{i}.pt")
    config = {"in_dim": 8, "hidden_dim": 16, "num_layers": 1}
    ckpt = tmp_path / "m.pt"
    torch.save({"model": AegisHeteroGNN(**config).state_dict(), "config": config}, ckpt)
    result = detect_poison(spectral_signature_scores(ckpt, [graphs]), poisoned, flag_fraction=0.1)
    assert result["recall"] == 1.0


def test_embed_is_what_the_classifier_sees():
    model = AegisHeteroGNN(in_dim=8, hidden_dim=16, num_layers=1).eval()
    x = torch.randn(3, 8)
    edges = {("NODE", r, "NODE"): torch.tensor([[0, 1], [1, 2]]) for r in ("AST", "CFG", "REACHING_DEF")}
    batch = torch.zeros(3, dtype=torch.long)
    assert torch.allclose(model(x, edges, batch), model.classifier(model.embed(x, edges, batch)))

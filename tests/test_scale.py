"""User journeys covered:

- As a developer re-scanning a large repository, I want unchanged functions
  reused from a cache so only edited code goes through Joern.
- As an operator, I want graph building spread over several Joern workers.
- As a reviewer, I want to see when a harmless-looking function calls risky code.
- As a user without a GPU, I want a smaller 8-bit model that gives nearly
  the same scores.
- As a CI system, I want to submit code to a queue over HTTP and poll for results.
"""
import time

import pytest
import torch
from test_build_training_graphs import _fake_joern_parsing_only

from aegis.model.hetero_gnn import AegisHeteroGNN
from aegis.pipeline.build_training_graphs import parallel_parse
from aegis.scanning.scale import GraphCache, call_graph, called_names, propagate_risk, quantize_for_cpu
from aegis.scanning.scanner import analyze

CODE = ("void sink(char *s) { char b[4]; strcpy(b, s); }\n"
        "void handler(char *input) { sink(input); }\n"
        "int add(int a, int b) { return a + b; }\n")


def _checkpoint(path):
    config = {"in_dim": 811, "hidden_dim": 16, "num_layers": 1}
    torch.save({"model": AegisHeteroGNN(**config).state_dict(), "config": config}, path)
    return path


def test_called_names_finds_calls_in_c_and_java():
    assert called_names("void f() { g(1); obj->h(); ns::k(); }", "cpp") == {"g", "h", "k"}
    assert called_names("class A { void f() { run(); x.go(); } }", "java") == {"run", "go"}


def test_risk_is_inherited_through_calls_with_the_path():
    functions = [{"function": "sink", "language": "cpp", "probability": 0.9,
                  "source": "void sink(char *s) { strcpy(b, s); }"},
                 {"function": "handler", "language": "cpp", "probability": 0.1,
                  "source": "void handler(char *i) { sink(i); }"},
                 {"function": "main", "language": "cpp", "probability": 0.05,
                  "source": "int main() { handler(0); }"},
                 {"function": "add", "language": "cpp", "probability": 0.05, "source": "int add() { return 1; }"}]
    edges = call_graph(functions)
    assert edges == {0: set(), 1: {0}, 2: {1}, 3: set()}
    propagate_risk(functions, edges, decay=0.8)
    assert functions[1]["context_risk"] == pytest.approx(0.72) and functions[1]["risk_path"] == ["handler", "sink"]
    assert functions[2]["context_risk"] == pytest.approx(0.8 ** 2 * 0.9)
    assert functions[3]["context_risk"] == 0.0 and functions[3]["risk_path"] == []


def test_recursion_does_not_loop_forever():
    functions = [{"function": "f", "language": "cpp", "probability": 0.5, "source": "void f() { f(); g(); }"},
                 {"function": "g", "language": "cpp", "probability": 0.4, "source": "void g() { f(); }"}]
    propagate_risk(functions, call_graph(functions))
    assert functions[0]["risk_path"] == ["f", "g"]


def test_second_scan_reuses_cached_graphs_and_only_parses_edited_functions(tmp_path):
    code = tmp_path / "a.c"
    code.write_text(CODE)
    ckpt = _checkpoint(tmp_path / "m.pt")
    calls = []
    runner = _fake_joern_parsing_only({"0.cpp", "1.cpp", "2.cpp"}, calls)
    first = analyze([code], {"cpp": ckpt}, tmp_path / "w", device="cpu", runner=runner, cache_dir=tmp_path / "c")
    assert first["stats"] == {"cached": 0, "parsed": 3} and len(first["functions"]) == 3
    joern_runs = len(calls)

    second = analyze([code], {"cpp": ckpt}, tmp_path / "w", device="cpu", runner=runner, cache_dir=tmp_path / "c")
    assert second["stats"] == {"cached": 3, "parsed": 0} and len(calls) == joern_runs  # Joern never ran
    assert [f["probability"] for f in second["functions"]] == [f["probability"] for f in first["functions"]]

    code.write_text(CODE.replace("return a + b;", "return a - b;"))
    third = analyze([code], {"cpp": ckpt}, tmp_path / "w", device="cpu", runner=runner, cache_dir=tmp_path / "c")
    assert third["stats"] == {"cached": 2, "parsed": 1}


def test_scan_reports_inherited_risk(tmp_path):
    code = tmp_path / "a.c"
    code.write_text(CODE)
    runner = _fake_joern_parsing_only({"0.cpp", "1.cpp", "2.cpp"}, [])
    result = analyze([code], {"cpp": _checkpoint(tmp_path / "m.pt")}, tmp_path / "w", device="cpu", runner=runner)
    handler = next(f for f in result["functions"] if f["function"] == "handler")
    assert handler["risk_path"] == ["handler", "sink"] and handler["context_risk"] > 0


def test_cache_key_changes_with_code_and_language(tmp_path):
    cache = GraphCache(tmp_path)
    assert GraphCache.key("cpp", "x") != GraphCache.key("cpp", "y") != GraphCache.key("java", "y")
    cache.put("k", {"g": 1})
    assert cache.get("k") == {"g": 1} and cache.get("missing") is None


def test_parallel_parse_uses_separate_workspaces_and_merges_results(tmp_path):
    from aegis.embeddings.codebert_embedder import load_model
    files = []
    for i in range(4):
        (tmp_path / f"{i}.cpp").write_text(f"int f{i}() {{ return {i}; }}")
        files.append(tmp_path / f"{i}.cpp")
    calls = []
    graphs = parallel_parse(files, tmp_path / "ws", *load_model("cpu"),
                            runner=_fake_joern_parsing_only({f"{i}.cpp" for i in range(4)}, calls), workers=2)
    assert sorted(graphs) == ["0.cpp", "1.cpp", "2.cpp", "3.cpp"]
    assert calls.count("joern-parse") == 2  # one batch per worker


def test_quantized_model_scores_close_to_the_full_model():
    torch.manual_seed(0)
    model = AegisHeteroGNN(in_dim=32, hidden_dim=64, num_layers=2).eval()
    small = quantize_for_cpu(model)
    x = torch.randn(10, 32)
    edges = {("NODE", r, "NODE"): torch.tensor([list(range(9)), list(range(1, 10))])
             for r in ("AST", "CFG", "REACHING_DEF")}
    batch = torch.zeros(10, dtype=torch.long)
    with torch.no_grad():
        full = torch.softmax(model(x, edges, batch), dim=1)
        quant = torch.softmax(small(x, edges, batch), dim=1)
    assert torch.allclose(full, quant, atol=0.05)


def _wait(client, job):
    for _ in range(200):
        body = client.get(f"/scans/{job}").json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.1)
    return body


def test_service_queues_a_scan_and_returns_results_without_server_paths(tmp_path):
    from fastapi.testclient import TestClient

    from aegis.service.api import create_app
    runner = _fake_joern_parsing_only({"0.cpp", "1.cpp", "2.cpp"}, [])
    client = TestClient(create_app({"cpp": _checkpoint(tmp_path / "m.pt")}, runner=runner, device="cpu"))
    assert client.get("/health").json()["models"] == ["cpp"]
    assert client.post("/scans", json={"files": {"notes.txt": "hi"}}).status_code == 400

    job = client.post("/scans", json={"files": {"src/a.c": CODE}}).json()["id"]
    body = _wait(client, job)
    assert body["status"] == "done", body
    assert {f["file"] for f in body["result"]["functions"]} == {"src/a.c"}
    assert client.get("/scans/nope").status_code == 404


def test_service_refuses_a_path_that_escapes_its_folder(tmp_path):
    from fastapi.testclient import TestClient

    from aegis.service.api import create_app
    client = TestClient(create_app({"cpp": _checkpoint(tmp_path / "m.pt")},
                                   runner=_fake_joern_parsing_only(set(), []), device="cpu"))
    job = client.post("/scans", json={"files": {"../../escape.c": "int f() { return 0; }"}}).json()["id"]
    body = _wait(client, job)
    assert body["status"] == "failed" and "unsafe" in body["error"]
    assert not (tmp_path.parent / "escape.c").exists()

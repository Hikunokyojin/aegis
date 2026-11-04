"""User journeys covered:

- As the operator of a free shared server, I want scans to run one at a time
  and each visitor to wait between scans, so nobody can exhaust it.
- As the operator, I want oversized inputs refused before expensive work.
- As a user, I want a tampered model file refused instead of loaded.
"""
import pytest
import torch

from aegis.evaluation.compare import ModelIntegrityError, load_trained_model
from aegis.languages import SOURCE_LANGUAGES, collect_source_files
from aegis.model.hetero_gnn import AegisHeteroGNN
from aegis.scanning.scanner import ScanTooLargeError, analyze
from aegis.training.trainer import sha256_of
from aegis.ui.analysis import RateLimitedError, ScanGate


def _checkpoint(path):
    config = {"in_dim": 811, "hidden_dim": 16, "num_layers": 1}
    torch.save({"model": AegisHeteroGNN(**config).state_dict(), "config": config}, path)
    return path


def test_gate_makes_a_visitor_wait_between_scans_but_not_others():
    now = [100.0]
    gate = ScanGate(cooldown_seconds=30, clock=lambda: now[0])
    with gate.slot("alice"):
        pass
    with pytest.raises(RateLimitedError):
        with gate.slot("alice"):
            pass
    with gate.slot("bob"):
        pass
    now[0] += 31
    assert gate.wait_time("alice") == 0
    with gate.slot("alice"):
        pass


def test_too_many_functions_are_refused_before_any_model_loads(tmp_path):
    code = tmp_path / "many.c"
    code.write_text("\n".join(f"int f{i}(void) {{ return {i}; }}" for i in range(5)))

    def never_called(*a, **k):
        raise AssertionError("Joern must not run for a refused scan")
    with pytest.raises(ScanTooLargeError):
        analyze([code], {"cpp": tmp_path / "missing.pt"}, tmp_path / "w", device="cpu",
                runner=never_called, max_functions=4)


def test_model_matching_its_checksum_loads(tmp_path):
    ckpt = _checkpoint(tmp_path / "m.pt")
    (tmp_path / "m.pt.sha256").write_text(f"{sha256_of(ckpt)}  m.pt\n")
    assert isinstance(load_trained_model(ckpt, "cpu"), AegisHeteroGNN)


def test_tampered_model_is_refused(tmp_path):
    ckpt = _checkpoint(tmp_path / "m.pt")
    (tmp_path / "m.pt.sha256").write_text("0" * 64 + "  m.pt\n")
    with pytest.raises(ModelIntegrityError):
        load_trained_model(ckpt, "cpu")


def test_model_file_that_would_run_code_is_refused(tmp_path):
    class Evil:
        def __reduce__(self):
            return (print, ("pwned",))
    bad = tmp_path / "evil.pt"
    torch.save({"model": {}, "config": {}, "x": Evil()}, bad)
    with pytest.raises(Exception, match="(?i)weights"):
        load_trained_model(bad, "cpu")


def test_one_language_table_includes_headers(tmp_path):
    assert SOURCE_LANGUAGES[".h"] == SOURCE_LANGUAGES[".hpp"] == "cpp"
    (tmp_path / "a.h").write_text("")
    assert [p.name for p in collect_source_files([tmp_path])] == ["a.h"]

"""User journeys covered:

- As a Python, JavaScript or Go developer, I want my functions found, turned
  into graphs and checked, and told honestly that no model is trained for my
  language yet instead of being given a made-up score.
- As someone who found a bug in a public repository, I want to be guided to
  report it privately, following the project's own security policy.
"""
import io
import zipfile

import pytest

from aegis.languages import SOURCE_LANGUAGES, language_of
from aegis.parsing.function_extractor import WRAPPER_LINES, extract_functions, to_standalone
from aegis.ui.analysis import detect_language, disclosure_guidance, safe_extract_zip

PY = "import os\n\nclass Store:\n    def load(self, name):\n        return open(name).read()\n\ndef run(cmd):\n    os.system(cmd)\n"
JS = ("const fs = require('fs');\nfunction read(p) {\n  return fs.readFileSync(p);\n}\n"
      "class Api {\n  handle(req) {\n    eval(req.body);\n  }\n}\n")
GO = ("package main\n\nimport \"os/exec\"\n\nfunc Run(cmd string) {\n\texec.Command(\"sh\", \"-c\", cmd).Run()\n}\n\n"
      "func (s *Server) Serve() {\n}\n")


@pytest.mark.parametrize("source,language,expected", [
    (PY, "python", [("load", 4), ("run", 7)]),
    (JS, "javascript", [("read", 2), ("handle", 6)]),
    (GO, "go", [("Run", 5), ("Serve", 9)]),
])
def test_functions_are_extracted_with_names_and_lines(source, language, expected):
    assert [(f.name, f.start_line) for f in extract_functions(source, language)] == expected


@pytest.mark.parametrize("source,language", [(PY, "python"), (JS, "javascript"), (GO, "go")])
def test_standalone_files_put_the_function_exactly_wrapper_lines_down(source, language):
    from tree_sitter import Parser

    from aegis.parsing.function_extractor import _PARSERS
    for fn in extract_functions(source, language):
        standalone = to_standalone(fn.source, language)
        first_line = standalone.splitlines()[WRAPPER_LINES[language]]
        assert first_line.strip() == fn.source.splitlines()[0].strip()
        assert not _PARSERS[language].parse(standalone.encode()).root_node.has_error
        assert isinstance(Parser, type)


def test_python_methods_are_dedented_so_they_parse_on_their_own():
    method = extract_functions(PY, "python")[0]
    assert to_standalone(method.source, "python").startswith("def load(self, name):\n    return")


def test_new_extensions_map_to_languages():
    assert [language_of(__import__("pathlib").Path(n)) for n in ("a.py", "b.mjs", "c.go", "d.jsx")] == \
        ["python", "javascript", "go", "javascript"]
    assert ".java" in SOURCE_LANGUAGES and ".c" in SOURCE_LANGUAGES


@pytest.mark.parametrize("code,language", [
    ("def handler(request):\n    return eval(request.args['q'])\n", "python"),
    ("package main\nfunc main() {}\n", "go"),
    ("const x = require('x');\nconsole.log(x)\n", "javascript"),
    ("#include <stdio.h>\nint main() { printf(\"hi\"); }", "cpp"),
    ("public class A { public static void main(String[] a) { System.out.println(1); } }", "java"),
])
def test_pasted_code_language_is_detected(code, language):
    assert detect_language(code) == language


def test_languages_without_a_model_are_reported_not_scored(tmp_path):
    import torch

    from aegis.model.hetero_gnn import AegisHeteroGNN
    from aegis.scanning.scanner import analyze
    (tmp_path / "app.py").write_text(PY)
    config = {"in_dim": 811, "hidden_dim": 16, "num_layers": 1}
    torch.save({"model": AegisHeteroGNN(**config).state_dict(), "config": config}, tmp_path / "m.pt")

    def never(*a, **k):
        raise AssertionError("no Joern run for a language without a model")
    result = analyze([tmp_path / "app.py"], {"cpp": tmp_path / "m.pt"}, tmp_path / "w", device="cpu", runner=never)
    assert result["functions"] == []
    assert {u["function"] for u in result["unparsed"]} == {"load", "run"}
    assert all(u["reason"] == "no trained model for python" for u in result["unparsed"])


def _zip(files: dict[str, str]):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in files.items():
            zf.writestr(name, text)
    return buf.getvalue()


def test_security_policy_is_kept_from_a_downloaded_repository_but_not_returned_as_source(tmp_path):
    archive = tmp_path / "r.zip"
    archive.write_bytes(_zip({"repo-main/src/a.c": "int f(){return 0;}", "repo-main/.github/SECURITY.md": "Email sec@x",
                              "repo-main/README.md": "hi"}))
    written = safe_extract_zip(archive, tmp_path / "out", keep_names=frozenset({"security.md"}))
    assert [p.name for p in written] == ["a.c"]
    assert (tmp_path / "out" / "repo-main" / ".github" / "SECURITY.md").exists()
    assert not (tmp_path / "out" / "repo-main" / "README.md").exists()


def test_disclosure_guidance_points_to_the_policy_and_the_private_report_form(tmp_path):
    (tmp_path / "repo-main" / ".github").mkdir(parents=True)
    (tmp_path / "repo-main" / ".github" / "SECURITY.md").write_text("Report to security@example.org")
    g = disclosure_guidance(tmp_path, "octo", "demo")
    assert g["policy_file"].replace("\\", "/") == "repo-main/.github/SECURITY.md"
    assert "security@example.org" in g["policy_text"]
    assert g["private_report_url"] == "https://github.com/octo/demo/security/advisories/new"
    assert any("public issue" in s for s in g["steps"])


def test_disclosure_guidance_without_a_policy_still_gives_the_private_route(tmp_path):
    g = disclosure_guidance(tmp_path, "octo", "demo")
    assert g["policy_file"] is None and g["private_report_url"].endswith("/security/advisories/new")

"""Aegis web UI (Streamlit). Start it with `aegis-ui`, then open the printed
local address in a browser.

Paste code, upload files or a .zip, or give a GitHub link; Aegis splits the
code into functions, scores each one with the trained model for its language,
highlights the lines that drove each score, optionally runs Semgrep and
CodeQL on the same code for comparison, and draws a function's code graph.
All analysis logic lives in aegis.ui.analysis; this file only draws it.
"""
import html
import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st

from aegis.ui.analysis import (
    SOURCE_LANGUAGES,
    analyze,
    attach_findings,
    detect_language,
    download_github_repo,
    graph_to_dot,
    run_codeql,
    run_semgrep,
    safe_extract_zip,
    write_pasted_code,
)

MODELS_DIR = Path(os.environ.get("AEGIS_MODELS_DIR", Path.home() / ".aegis" / "models"))
CPP_MODELS = {"BigVul (real-world C/C++)": "bigvul.pt", "Juliet C/C++ (synthetic test suite)": "juliet-cpp.pt"}
JAVA_MODEL = "juliet-java.pt"
EXAMPLE_C = """#include <string.h>
#include <stdio.h>

void copy_input(const char *user_input) {
    char buffer[16];
    strcpy(buffer, user_input);   /* no length check */
    printf("%s\\n", buffer);
}

int add(int a, int b) {
    return a + b;
}
"""

st.set_page_config(page_title="Aegis — vulnerability scanner", page_icon="🛡️", layout="wide")


def heat_html(source: str, start_line: int, line_scores: dict[int, float], findings: list[dict]) -> str:
    """The function's code with each line shaded by its contribution to the
    score, and tool findings marked in the margin."""
    marks = {}
    for f in findings:
        marks.setdefault(f["line"], []).append(f["tool"][0].upper())
    rows = []
    for i, text in enumerate(source.splitlines(), start=start_line):
        score = line_scores.get(i, 0.0)
        tag = "".join(sorted(set(marks.get(i, [])))) or "&nbsp;"
        rows.append(
            f'<div style="background: rgba(220,38,38,{0.55 * score:.2f}); white-space: pre; font-family: monospace;'
            f' font-size: 13px; padding: 0 6px;"><span style="color:#888; display:inline-block; width:3.5em">{i}</span>'
            f'<span style="color:#b45309; display:inline-block; width:1.6em">{tag}</span>{html.escape(text)}</div>')
    return '<div style="border:1px solid #ddd; border-radius:6px; overflow-x:auto">' + "".join(rows) + "</div>"


def collect_files(work: Path, mode: str, pasted: str, language: str, uploads, github_url: str) -> list[Path]:
    inputs = work / "input"
    shutil.rmtree(inputs, ignore_errors=True)
    inputs.mkdir(parents=True)
    if mode == "Paste code":
        lang = detect_language(pasted) if language == "Auto-detect" else ("java" if language == "Java" else "cpp")
        return [write_pasted_code(pasted, lang, inputs)]
    if mode == "Upload files":
        files = []
        for up in uploads or []:
            name = Path(up.name).name
            target = inputs / name
            target.write_bytes(up.getbuffer())
            if name.lower().endswith(".zip"):
                files += safe_extract_zip(target, inputs / Path(name).stem)
                target.unlink()
            elif Path(name).suffix.lower() in SOURCE_LANGUAGES:
                files.append(target)
        return files
    return download_github_repo(github_url, inputs)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    cpp_choice = st.selectbox("Model for C/C++ code", list(CPP_MODELS))
    model_paths = {}
    for language, filename in (("cpp", CPP_MODELS[cpp_choice]), ("java", JAVA_MODEL)):
        path = MODELS_DIR / filename
        if path.exists():
            model_paths[language] = path
    st.caption(f"Models folder: `{MODELS_DIR}`")
    st.write(("✅" if "cpp" in model_paths else "❌") + " C/C++ model", "  \n",
             ("✅" if "java" in model_paths else "❌") + " Java model")
    threshold = st.slider("Flag functions with probability ≥", 0.05, 0.95, 0.5, 0.05)
    explain = st.checkbox("Explain scores (highlight lines)", value=True)
    use_semgrep = st.checkbox("Compare with Semgrep", value=shutil.which("semgrep") is not None,
                              disabled=shutil.which("semgrep") is None)
    use_codeql = st.checkbox("Compare with CodeQL (slow)", value=False, disabled=shutil.which("codeql") is None)
    if shutil.which("joern-parse") is None:
        st.error("Joern is not on PATH; scans will fail. See the run guide.")

# ---------------------------------------------------------------- input
st.title("🛡️ Aegis")
st.write("Finds functions likely to contain security vulnerabilities in **C/C++** and **Java**, using a graph neural "
         "network trained on code property graphs (syntax, control flow and data flow).")

mode = st.radio("Input", ["Paste code", "Upload files", "GitHub link"], horizontal=True)
pasted, language, uploads, github_url = "", "Auto-detect", None, ""
if mode == "Paste code":
    language = st.selectbox("Language", ["Auto-detect", "C/C++", "Java"])
    pasted = st.text_area("Code", value=EXAMPLE_C, height=280)
elif mode == "Upload files":
    uploads = st.file_uploader("C/C++/Java files or a .zip of a project", accept_multiple_files=True,
                               type=["c", "cc", "cpp", "cxx", "h", "hpp", "java", "zip"])
else:
    github_url = st.text_input("Public GitHub repository", placeholder="https://github.com/owner/repository")
    st.caption("The default branch is downloaded (max 50 MB) and its C/C++/Java files are scanned. "
               "Large repositories take long.")

if st.button("Scan", type="primary"):
    work = Path(tempfile.gettempdir()) / "aegis_ui"
    try:
        with st.status("Scanning…", expanded=True) as status:
            st.write("Collecting source files")
            files = collect_files(work, mode, pasted, language, uploads, github_url)
            if not files:
                raise ValueError("No C/C++/Java source files found in the input.")
            st.write(f"{len(files)} file(s). Building code graphs with Joern and scoring functions…")
            result = analyze(files, model_paths, work / "analysis", explain=explain)
            findings = []
            if use_semgrep:
                st.write("Running Semgrep…")
                findings += run_semgrep(files, work / "tools")
            if use_codeql:
                st.write("Running CodeQL…")
                findings += run_codeql(files, work / "tools")
            unmatched = attach_findings(result["functions"], findings)
            st.session_state["result"] = {**result, "unmatched": unmatched, "root": str(work / "input"),
                                          "tools": [t for t, on in (("Semgrep", use_semgrep), ("CodeQL", use_codeql))
                                                    if on]}
            status.update(label="Scan complete", state="complete")
    except Exception as exc:  # shown to the user rather than crashing the page
        st.error(f"Scan failed: {exc}")

# ---------------------------------------------------------------- results
result = st.session_state.get("result")
if result:
    fns, root = result["functions"], result["root"]
    def rel(p):
        return os.path.relpath(p, root)
    flagged = [f for f in fns if f["probability"] >= threshold]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Functions scored", len(fns))
    c2.metric("Flagged by Aegis", len(flagged))
    c3.metric("With tool findings", sum(1 for f in fns if f.get("findings")))
    c4.metric("Not analysed", len(result["unparsed"]))

    st.subheader("Functions ranked by risk")
    st.dataframe([{"Risk": round(f["probability"], 3), "Flag": "⚠️" if f["probability"] >= threshold else "",
                   "Function": f["function"], "Location": f"{rel(f['file'])}:{f['start_line']}",
                   "Semgrep": sum(x["tool"] == "semgrep" for x in f.get("findings", [])),
                   "CodeQL": sum(x["tool"] == "codeql" for x in f.get("findings", []))} for f in fns],
                 use_container_width=True, hide_index=True)
    st.caption("Risk is the model's probability that the function is vulnerable. In the code views, red shading shows "
               "how much each line pushed that probability up; S/C in the margin mark Semgrep/CodeQL findings.")

    for i, f in enumerate(fns):
        flag = "⚠️ " if f["probability"] >= threshold else ""
        label = f"{flag}{f['function']} — {f['probability']:.3f} — {rel(f['file'])}:{f['start_line']}"
        with st.expander(label, expanded=i == 0):
            st.markdown(heat_html(f["source"], f["start_line"], f["line_scores"], f.get("findings", [])),
                        unsafe_allow_html=True)
            if f.get("findings"):
                st.write("**Tool findings**")
                for x in f["findings"]:
                    st.write(f"- {x['tool']} · line {x['line']} · `{x['rule']}` — {x['message']}")
            elif result["tools"]:
                st.caption(f"No {' / '.join(result['tools'])} findings in this function.")
            if st.toggle("Show code graph", key=f"graph{i}"):
                st.caption("Grey: syntax (AST) · Blue: control flow (CFG) · Red: data flow (REACHING_DEF). "
                           "Darker nodes contributed more to the score.")
                st.graphviz_chart(graph_to_dot(f["graph"], f["line_scores"]), use_container_width=True)

    if result["unparsed"]:
        st.subheader("Not analysed")
        for u in result["unparsed"]:
            st.write(f"- `{u['function']}` at {rel(u['file'])}:{u['start_line']} — {u['reason']}")

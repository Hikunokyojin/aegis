"""A small HTTP scanning service with a job queue, so other programs (a CI
job, an editor plugin, the web app on another machine) can submit code and
collect results.

    POST /scans          {"files": {"path/name.c": "<code>", ...}}  -> 202 {"id": ...}
    GET  /scans/{id}     -> {"status": "queued"|"running"|"done"|"failed", "result"?, "error"?}
    GET  /health         -> models loaded, queue length

One worker thread takes jobs in order, so a burst of requests queues instead
of starting many Joern/JVM processes at once (the expensive part); the
queue and request sizes are capped. Submitted code is written to a private
temporary folder and deleted when its job ends, and only the last
`MAX_KEPT_RESULTS` results are kept in memory. Start it with `aegis-serve`.
"""
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import uuid
from collections import OrderedDict
from functools import partial
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from aegis.languages import language_of
from aegis.scanning.scanner import analyze

MAX_FILES = int(os.environ.get("AEGIS_MAX_FILES", 200))
MAX_TOTAL_BYTES = int(os.environ.get("AEGIS_MAX_BYTES", 5 * 1024 * 1024))
MAX_QUEUE = int(os.environ.get("AEGIS_MAX_QUEUE", 20))
MAX_FUNCTIONS = int(os.environ.get("AEGIS_MAX_FUNCTIONS", 200))
MAX_KEPT_RESULTS = 100
TOOL_TIMEOUT_SECONDS = int(os.environ.get("AEGIS_TOOL_TIMEOUT", 300))


class ScanRequest(BaseModel):
    files: dict[str, str] = Field(..., description="Relative file path -> source code")


def default_model_paths() -> dict[str, Path]:
    models = Path(os.environ.get("AEGIS_MODELS_DIR", Path.home() / ".aegis" / "models"))
    chosen = {"cpp": os.environ.get("AEGIS_CPP_MODEL", "bigvul.pt"), "java": "juliet-java.pt"}
    return {lang: models / name for lang, name in chosen.items() if (models / name).exists()}


def _report(result: dict) -> dict:
    """The JSON-safe part of an analysis (graphs and line maps are dropped or made serialisable)."""
    keep = ("file", "function", "language", "start_line", "end_line", "probability", "uncertain",
            "context_risk", "risk_path")
    return {"functions": [{**{k: f.get(k) for k in keep},
                           "line_scores": {str(k): v for k, v in f.get("line_scores", {}).items()}}
                          for f in result["functions"]],
            "unparsed": [{k: u[k] for k in ("file", "function", "start_line", "reason")} for u in result["unparsed"]],
            "stats": result.get("stats", {})}


def create_app(model_paths: dict[str, Path] | None = None, runner=None, cache_dir: Path | None = None,
               device: str | None = None) -> FastAPI:
    """Builds the service. `runner` and `device` are injectable for tests."""
    model_paths = default_model_paths() if model_paths is None else model_paths
    runner = runner or partial(subprocess.run, timeout=TOOL_TIMEOUT_SECONDS)
    jobs: OrderedDict[str, dict] = OrderedDict()
    pending: queue.Queue = queue.Queue(maxsize=MAX_QUEUE)
    lock = threading.Lock()

    def work() -> None:
        while True:
            job_id, files = pending.get()
            with lock:
                jobs[job_id]["status"] = "running"
            folder = Path(tempfile.mkdtemp(prefix="aegis_api_"))
            try:
                paths = []
                for name, code in files.items():
                    target = (folder / "input" / name).resolve()
                    if not target.is_relative_to((folder / "input").resolve()):
                        raise ValueError(f"unsafe file path: {name}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(code, encoding="utf-8")
                    paths.append(target)
                result = analyze(paths, model_paths, folder / "work", device=device, runner=runner,
                                 max_functions=MAX_FUNCTIONS, cache_dir=cache_dir)
                report = _report(result)
                prefix = str(folder / "input") + os.sep
                for item in report["functions"] + report["unparsed"]:
                    item["file"] = item["file"].removeprefix(prefix)  # never leak server paths
                update = {"status": "done", "result": report}
            except Exception as exc:  # reported to the client, the worker keeps serving
                update = {"status": "failed", "error": str(exc)}
            finally:
                shutil.rmtree(folder, ignore_errors=True)
            with lock:
                jobs[job_id].update(update)
                while len(jobs) > MAX_KEPT_RESULTS:
                    jobs.popitem(last=False)
            pending.task_done()

    threading.Thread(target=work, daemon=True).start()
    app = FastAPI(title="Aegis scanning service", version="1.0")

    @app.get("/health")
    def health():
        return {"models": sorted(model_paths), "queued": pending.qsize()}

    @app.post("/scans", status_code=202)
    def submit(request: ScanRequest):
        files = {k: v for k, v in request.files.items() if language_of(Path(k))}
        if not files:
            raise HTTPException(400, "No C/C++/Java files in the request.")
        if len(files) > MAX_FILES or sum(len(v.encode()) for v in files.values()) > MAX_TOTAL_BYTES:
            raise HTTPException(413, f"At most {MAX_FILES} files and {MAX_TOTAL_BYTES} bytes per scan.")
        job_id = uuid.uuid4().hex
        with lock:
            jobs[job_id] = {"status": "queued"}
        try:
            pending.put_nowait((job_id, files))
        except queue.Full as full:
            with lock:
                jobs.pop(job_id)
            raise HTTPException(503, "The scan queue is full; try again shortly.") from full
        return {"id": job_id}

    @app.get("/scans/{job_id}")
    def status(job_id: str):
        with lock:
            job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown or expired scan id.")
        return job

    app.state.pending = pending
    return app


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the Aegis scanning service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".aegis" / "cache")
    args = parser.parse_args()
    uvicorn.run(create_app(cache_dir=args.cache_dir), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

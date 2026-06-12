"""Local web server for the memory inspector (stdlib only, no web framework).

Run:  python demo/server.py            # http://127.0.0.1:8000
      python demo/server.py 8001       # custom port

Flow: the browser POSTs a query + options to /api/run; we run the
``DeepResearchPipeline`` (wired to a ``TracingMemoryManager``) in a background
thread; the page polls /api/runs/<id>/events for the live read->decide->write
trace and /api/runs/<id>/memory for the current memory state.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import llm  # noqa: E402
from memory_dr import MemoryStore  # noqa: E402
from pipeline import DeepResearchPipeline  # noqa: E402
from run_demo import QUESTION, load_dotenv  # noqa: E402
from search_client import SearchClient  # noqa: E402
from trace import Tracer, TracingMemoryManager, snapshot  # noqa: E402

HERE = os.path.dirname(__file__)
UI_PATH = os.path.join(HERE, "webui", "index.html")
TRACE_PATH = os.path.join(HERE, ".demo_trace.json")

_runs: Dict[str, "Run"] = {}
_runs_lock = threading.Lock()


class Run:
    def __init__(self, run_id: str, params: Dict[str, Any]) -> None:
        self.id = run_id
        self.params = params
        self.tracer = Tracer(dump_path=TRACE_PATH)
        self.memory: Optional[TracingMemoryManager] = None
        self.pipeline: Optional[DeepResearchPipeline] = None
        self.status = "running"
        self.error: Optional[str] = None
        self.report: Optional[str] = None
        self.config: Dict[str, Any] = {}

    def metrics(self) -> Dict[str, Any]:
        m: Dict[str, Any] = {"steps": 0, "searches_run": 0, "searches_skipped": 0}
        if self.pipeline is not None:
            m.update(
                steps=self.pipeline.steps,
                searches_run=self.pipeline.searches_run,
                searches_skipped=self.pipeline.searches_skipped,
            )
        if self.memory is not None:
            m["memory"] = self.memory.stats()
        return m


def _execute(run: Run) -> None:
    """Run the pipeline for one query; all observability flows via run.tracer."""
    p = run.params
    no_llm = bool(p.get("no_llm", False))
    want_mock = bool(p.get("mock", True))
    rounds = int(p.get("rounds", 3))
    page_size = int(p.get("page_size", 6))
    biz_types = [b.strip() for b in str(p.get("biz", "paper")).split(",") if b.strip()] or ["paper"]
    question = (p.get("query") or QUESTION).strip()

    try:
        try:
            client = SearchClient(mock=want_mock)
            effective_mock = want_mock
        except ValueError as e:
            run.tracer.record("info", {"event": "config", "fellback": True, "detail": str(e)})
            client = SearchClient(mock=True)
            effective_mock = True

        use_llm = not no_llm
        run.config = {
            "mock": effective_mock,
            "use_llm": use_llm,
            "llm_available": llm.llm_available() if use_llm else False,
            "base_url": os.environ.get("SEARCH_API_BASE_URL", ""),
            "model": os.environ.get("MEMORY_DR_MODEL", "") if use_llm else "",
            "biz_types": biz_types,
            "rounds": rounds,
            "question": question,
        }
        run.tracer.record("info", {"event": "start", **run.config})

        store = MemoryStore(path=None)  # per-run, in-memory
        run.memory = TracingMemoryManager(store=store, task_id="task-alpha", tracer=run.tracer)
        run.pipeline = DeepResearchPipeline(
            run.memory,
            client,
            use_llm=use_llm,
            biz_types=biz_types,
            page_size=page_size,
            tracer=run.tracer,
            verbose=False,
        )
        run.report = run.pipeline.run(question, rounds=rounds)
        run.status = "done"
        run.tracer.record("info", {"event": "done"})
    except Exception as e:  # surface any failure to the UI instead of dying silently
        run.status = "error"
        run.error = str(e)
        run.tracer.record("info", {"event": "error", "error": str(e)})


class Handler(BaseHTTPRequestHandler):
    server_version = "MemoryInspector/0.1"

    # --- helpers --------------------------------------------------------
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def log_message(self, *args: Any) -> None:  # silence default access logging
        return

    # --- routing --------------------------------------------------------
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            return self._serve_ui()
        parts = [p for p in path.split("/") if p]
        # /api/runs/<id>/events  |  /api/runs/<id>/memory
        if len(parts) == 4 and parts[:2] == ["api", "runs"]:
            run_id, leaf = parts[2], parts[3]
            run = _get_run(run_id)
            if run is None:
                return self._json({"error": "unknown run_id"}, 404)
            if leaf == "events":
                since = int((parse_qs(parsed.query).get("since", ["0"])[0]) or 0)
                events = run.tracer.events_since(since)
                return self._json(
                    {
                        "status": run.status,
                        "error": run.error,
                        "events": events,
                        "max_seq": run.tracer._seq,
                        "metrics": run.metrics(),
                        "config": run.config,
                        "report": run.report if run.status == "done" else None,
                    }
                )
            if leaf == "memory":
                snap = snapshot(run.memory) if run.memory is not None else {}
                return self._json({"memory": snap, "metrics": run.metrics()})
        return self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/run":
            return self._json({"error": "not found"}, 404)
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            params = json.loads(raw.decode("utf-8") or "{}")
        except (ValueError, json.JSONDecodeError) as e:
            return self._json({"error": f"bad request body: {e}"}, 400)

        run_id = uuid.uuid4().hex[:12]
        run = Run(run_id, params if isinstance(params, dict) else {})
        with _runs_lock:
            _runs[run_id] = run
        threading.Thread(target=_execute, args=(run,), daemon=True).start()
        return self._json({"run_id": run_id})

    def _serve_ui(self) -> None:
        try:
            with open(UI_PATH, "rb") as fh:
                body = fh.read()
        except OSError:
            return self._send(500, b"UI not found (demo/webui/index.html)", "text/plain; charset=utf-8")
        return self._send(200, body, "text/html; charset=utf-8")


def _get_run(run_id: str) -> Optional[Run]:
    with _runs_lock:
        return _runs.get(run_id)


def main() -> None:
    load_dotenv()  # pick up SEARCH_API_* and ANTHROPIC_* before any run
    port = int(os.environ.get("MEMORY_DR_PORT", "8000"))
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    # Defaults to localhost (safe). On a remote/dev box, either tunnel it (see
    # hint below) or set MEMORY_DR_HOST=0.0.0.0 to bind all interfaces.
    host = os.environ.get("MEMORY_DR_HOST", "127.0.0.1")
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Memory inspector running at http://{host}:{port}  (Ctrl-C to stop)")
    if host in ("127.0.0.1", "localhost"):
        print(
            f"  on a remote/dev box? from your laptop run:\n"
            f"    ssh -N -L {port}:localhost:{port} <user>@<host>\n"
            f"  then open http://localhost:{port}"
        )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping…")
        httpd.shutdown()


if __name__ == "__main__":
    main()

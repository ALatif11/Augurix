"""Run generated strategy code in a locked-down Docker container. Never run it on the host.

The container has no network, no data files and a read-only filesystem. Bars go in
on stdin and positions come back on stdout, so the strategy can only see what it is handed.
"""
import io
import json
import subprocess
import tempfile
import uuid
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

IMAGE = "augurix-sandbox"
ROOT = Path(__file__).parent.resolve()


def pack(frames):
    """{symbol: bars DataFrame} -> Parquet bytes for the container's stdin."""
    table = pa.Table.from_pandas(pd.concat([df.assign(symbol=s) for s, df in frames.items()]),
                                 preserve_index=False)
    buf = io.BytesIO()
    pq.write_table(table.replace_schema_metadata(None), buf)  # drop host-pandas metadata the older pandas inside cannot read
    return buf.getvalue()


def run(code, packed, timeout=60, detector=True):
    """Return the harness result dict: positions, or {"error": kind, "detail": text}.

    detector=False skips the look-ahead test, for re-running code that was already judged (the dashboard replay).
    """
    name = f"augurix-{uuid.uuid4().hex[:12]}"
    with tempfile.TemporaryDirectory() as job:
        Path(job, "strategy.py").write_text(code, encoding="utf-8")
        cmd = ["docker", "run", "--rm", "-i", "--name", name, "--network", "none",
               "--cpus", "1", "--memory", "1g", "--memory-swap", "1g", "--pids-limit", "128",
               "--read-only", "--tmpfs", "/tmp",
               "-v", f"{job}:/job:ro", "-v", f"{ROOT / 'harness.py'}:/harness.py:ro"]
        if detector and (ROOT / "lookahead.py").exists():
            cmd += ["-v", f"{ROOT / 'lookahead.py'}:/lookahead.py:ro"]
        cmd += [IMAGE, "timeout", str(timeout), "python", "/harness.py"]
        try:
            p = subprocess.run(cmd, input=packed, capture_output=True, timeout=timeout + 30)
        except subprocess.TimeoutExpired:
            # Killing the docker client does not stop the container.
            subprocess.run(["docker", "kill", name], capture_output=True)
            return {"error": "timeout", "detail": f"no result after {timeout + 30}s"}
    if p.returncode == 124:
        return {"error": "timeout", "detail": f"ran longer than {timeout}s"}
    # 125: docker itself failed (engine down, image missing). 0xC0000000 and up: Windows could not start the docker
    # client at all (pilot5's last five attempts, logged while the session was shutting down). Neither is the
    # strategy's fault, so neither may be logged as an attempt or sent to the coder as a crash.
    if p.returncode == 125 or p.returncode >= 0xC0000000:
        raise RuntimeError("docker could not start the sandbox: " + p.stderr.decode(errors="replace")[-300:])
    try:
        return json.loads(p.stdout)
    except ValueError:
        return {"error": "crash",
                "detail": f"exit code {p.returncode}: {p.stderr.decode(errors='replace')[-800:]}"}

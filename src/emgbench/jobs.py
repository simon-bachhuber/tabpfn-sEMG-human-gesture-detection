"""Run long experiments with a durable log and exit status."""

import os
import subprocess
import sys

from .common import ROOT, atomic_json, read_json, utc_now


def launch(name, command):
    if not name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Job names must be alphanumeric, with optional - or _")
    directory = ROOT / "artifacts/jobs" / name
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / "status.json"
    if status_path.exists():
        old = read_json(status_path)
        if old.get("status") in {"starting", "running"}:
            try:
                os.kill(old["pid"], 0)
            except ProcessLookupError:
                pass
            else:
                raise ValueError(f"Job {name} is still running (PID {old['pid']})")
    atomic_json(directory / "command.json", {"command": command})
    with (directory / "output.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "emgbench.jobs", name], cwd=ROOT,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        )
    return {"name": name, "pid": process.pid, "log": str(directory / "output.log")}


def worker(name):
    directory = ROOT / "artifacts/jobs" / name
    command = read_json(directory / "command.json")["command"]
    status = {"pid": os.getpid(), "command": command, "started_at": utc_now(), "status": "running"}
    atomic_json(directory / "status.json", status)
    try:
        result = subprocess.run(command, cwd=ROOT, check=False)
        status.update(exit_code=result.returncode, status="completed" if result.returncode == 0 else "failed")
    except Exception as error:
        status.update(status="failed", error=str(error))
    status["finished_at"] = utc_now()
    atomic_json(directory / "status.json", status)


if __name__ == "__main__":
    worker(sys.argv[1])

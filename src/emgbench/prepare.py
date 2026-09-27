"""Acquire and preprocess the frozen cohort, overlapping downloads with CPU work."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from time import perf_counter

from .common import ROOT, atomic_json, digest, settings, utc_now
from .data import catalog, download_asset
from .processing import process_recording
from .reference import ensure_reference
from .training_replay import training_subjects


def prepare(workers=6, download_workers=3):
    if workers < 1 or download_workers < 1:
        raise ValueError("Worker counts must be positive")
    ensure_reference()
    config = settings()
    listing = catalog()
    expected = set(config["train_subjects"] + config["test_subjects"])
    actual = {r["subject"] for r in listing["recordings"]}
    if actual != expected or len(listing["recordings"]) != 264:
        raise ValueError("The frozen 44-person cohort does not match the public manifest")
    root = ROOT / "artifacts/preparation"
    state_path = root / "preparation.json"
    state = {
        "configuration": config, "configuration_sha256": digest(config),
        "started_at": utc_now(), "status": "running", "expected_recordings": 264,
        "downloaded": 0, "processed": 0, "workers": workers,
        "download_workers": download_workers, "errors": [],
    }
    atomic_json(state_path, state)
    started = perf_counter()
    process_jobs, reports = {}, []
    with ProcessPoolExecutor(max_workers=workers) as processors, ThreadPoolExecutor(max_workers=download_workers) as downloads:
        fetch_jobs = {}
        for record in listing["recordings"]:
            path = ROOT / "data/raw/hdf5" / record["hdf5"]["name"]
            fetch_jobs[downloads.submit(download_asset, record["hdf5"], path)] = (record, path)
        for future in as_completed(fetch_jobs):
            record, path = fetch_jobs[future]
            try:
                future.result()
                state["downloaded"] += 1
                process_jobs[processors.submit(process_recording, path)] = record["recording"]
            except Exception as error:
                state["errors"].append({"recording": record["recording"], "stage": "download", "error": str(error)})
            finished = [job for job in process_jobs if job.done()]
            for job in finished:
                stem = process_jobs.pop(job)
                try:
                    reports.append(job.result())
                    state["processed"] += 1
                except Exception as error:
                    state["errors"].append({"recording": stem, "stage": "processing", "error": str(error)})
            state["updated_at"] = utc_now()
            atomic_json(state_path, state)

        first_days = {
            subject: min(r["date"] for r in listing["recordings"] if r["subject"] == subject)
            for subject in config["test_subjects"] + training_subjects()
        }
        video_jobs = []
        for record in listing["recordings"]:
            if record["subject"] in first_days and record["date"] == first_days[record["subject"]] and record["trajectory"] == "sequential":
                asset = record["video"]
                if asset is None:
                    raise ValueError(f"Missing replay video: {record['recording']}")
                video_jobs.append(downloads.submit(download_asset, asset, ROOT / "data/raw/video-576p" / asset["name"]))
        for job in video_jobs:
            job.result()
        state["held_out_videos"] = len(config["test_subjects"])
        state["training_videos"] = len(training_subjects())
        for job in as_completed(process_jobs):
            stem = process_jobs[job]
            try:
                reports.append(job.result())
                state["processed"] += 1
            except Exception as error:
                state["errors"].append({"recording": stem, "stage": "processing", "error": str(error)})
            state["updated_at"] = utc_now()
            atomic_json(state_path, state)
    state.update(
        status="failed" if state["errors"] else "completed",
        finished_at=utc_now(), wall_seconds=perf_counter() - started,
        windows=sum(r["windows"] for r in reports),
        recorded_filter_task_seconds=sum(r["filter_seconds"] for r in reports),
        feature_task_seconds=sum(r["feature_seconds"] for r in reports),
    )
    atomic_json(state_path, state)
    if state["errors"] or state["processed"] != 264:
        raise RuntimeError(f"Cohort preparation incomplete; inspect {state_path}")
    print(f"Cohort ready: {state['processed']} recordings, {state['windows']} windows, {state['wall_seconds'] / 60:.1f} minutes", flush=True)
    return state

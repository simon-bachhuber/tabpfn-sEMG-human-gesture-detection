"""Manifest-driven acquisition from the current public Nextcloud share."""

import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote, unquote

import httpx

from .common import ROOT, atomic_json, read_json, sha256, utc_now

BASE_URL = "https://chmura.put.poznan.pl/public.php/dav/files/45NY5snj0U4tgQz/"
RECORD_PATTERN = re.compile(
    r"emg_gestures-(?P<subject>\d{2})-(?P<trajectory>repeats_long|repeats_short|sequential)-"
    r"(?P<date>\d{4}-\d{2}-\d{2})-(?P<time>\d{2}-\d{2}-\d{2}-\d{3})"
)
TRAJECTORIES = {"repeats_long", "repeats_short", "sequential"}


def parse_record(stem):
    match = RECORD_PATTERN.fullmatch(stem)
    if match is None:
        raise ValueError(f"Unexpected gesture recording name: {stem}")
    return {"recording": stem, **match.groupdict()}


def remote_listing(folder):
    body = (
        '<d:propfind xmlns:d="DAV:"><d:prop><d:getcontentlength/>'
        '<d:getetag/><d:getlastmodified/></d:prop></d:propfind>'
    )
    response = httpx.request(
        "PROPFIND", BASE_URL + folder + "/", content=body,
        headers={"Depth": "1", "Content-Type": "application/xml"},
        timeout=90, follow_redirects=True,
    )
    response.raise_for_status()
    items = {}
    for node in ET.fromstring(response.content).findall("{DAV:}response"):
        href = node.findtext("{DAV:}href", default="")
        name = unquote(href.rsplit("/", 1)[-1])
        if not name.startswith("emg_gestures-"):
            continue
        parse_record(Path(name).stem)
        items[name] = {
            "name": name,
            "size_bytes": int(node.findtext(".//{DAV:}getcontentlength", default="0")),
            "etag": node.findtext(".//{DAV:}getetag"),
            "last_modified": node.findtext(".//{DAV:}getlastmodified"),
            "url": BASE_URL + folder + "/" + quote(name),
        }
    return items


def catalog():
    response = httpx.get(BASE_URL + "records.txt", timeout=60, follow_redirects=True)
    response.raise_for_status()
    records = sorted({line.strip() for line in response.text.splitlines() if line.startswith("emg_gestures-")})
    parsed = [parse_record(stem) for stem in records]
    groups = defaultdict(list)
    for record in parsed:
        groups[(record["subject"], record["date"])].append(record["trajectory"])
    for key, trajectories in groups.items():
        if len(trajectories) != 3 or set(trajectories) != TRAJECTORIES:
            raise ValueError(f"Incomplete/duplicate trajectory group {key}: {trajectories}")
    hdf = remote_listing("Data-HDF5")
    videos = remote_listing("Video-576p")
    for record in parsed:
        stem = record["recording"]
        if stem + ".hdf5" not in hdf:
            raise ValueError(f"Manifest recording missing from HDF5 listing: {stem}")
        record["hdf5"] = hdf[stem + ".hdf5"]
        record["video"] = videos.get(stem + ".mp4")
    result = {
        "retrieved_at": utc_now(), "source": BASE_URL,
        "dataset_license": "CC-BY-NC-4.0", "published_checksums_available": False,
        "recordings": parsed,
    }
    atomic_json(ROOT / "artifacts/catalog.json", result)
    return result


def download_asset(asset, destination):
    """Resume our partial file; validate size and retain a content hash."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = destination.with_suffix(destination.suffix + ".metadata.json")
    if destination.exists():
        if not metadata_path.exists():
            raise ValueError(f"Existing untracked input: {destination}")
        metadata = read_json(metadata_path)
        if destination.stat().st_size != asset["size_bytes"] or sha256(destination) != metadata["sha256"]:
            raise ValueError(f"Input integrity mismatch: {destination}")
        if metadata["etag"] != asset["etag"]:
            raise ValueError(f"Remote source revision changed: {destination.name}")
        print(f"Verified cached {destination.name}", flush=True)
        return metadata

    partial = destination.with_suffix(destination.suffix + ".part")
    partial_metadata = partial.with_suffix(partial.suffix + ".json")
    if partial.exists():
        if not partial_metadata.exists() or read_json(partial_metadata).get("etag") != asset["etag"]:
            raise ValueError(f"Partial download has unknown/different source revision: {partial}")
    atomic_json(partial_metadata, asset)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > asset["size_bytes"]:
        raise ValueError(f"Oversized partial download: {partial}")
    if offset < asset["size_bytes"]:
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        print(f"Downloading {destination.name} ({asset['size_bytes'] / 1e6:.1f} MB)", flush=True)
        with httpx.stream("GET", asset["url"], headers=headers, timeout=120, follow_redirects=True) as response:
            response.raise_for_status()
            if response.status_code == 206:
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {offset}-"):
                    raise ValueError(f"Unexpected range response: {content_range}")
            elif response.status_code == 200:
                offset = 0  # Server does not honor ranges; restart our partial file.
            else:
                raise ValueError(f"Unexpected download status: {response.status_code}")
            with partial.open("ab" if offset else "wb") as stream:
                for chunk in response.iter_bytes(1024 * 1024):
                    stream.write(chunk)
    if partial.stat().st_size != asset["size_bytes"]:
        raise ValueError(f"Incomplete download: {partial}")
    metadata = {**asset, "sha256": sha256(partial), "downloaded_at": utc_now()}
    partial.replace(destination)
    atomic_json(metadata_path, metadata)
    partial_metadata.unlink()
    print(f"Complete {destination.name}", flush=True)
    return metadata

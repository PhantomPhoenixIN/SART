#!/usr/bin/env python3
"""Resume-safe retrieval of the official XLCoST Google Drive archive."""

from __future__ import annotations

import hashlib
import html.parser
import json
import os
import re
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

FILE_ID = "1Cp3vFITRaUEJwPoeI_uv0cC6KVyvDc4F"
SOURCE_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
USER_AGENT = "SART-CT reproducibility artifact downloader/0.1"


class FormParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.action: str | None = None
        self.hidden: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "form" and values.get("action"):
            self.action = values["action"]
        if tag == "input" and values.get("type") == "hidden":
            name, value = values.get("name"), values.get("value")
            if name and value is not None:
                self.hidden[name] = value


def request_for_download() -> urllib.request.Request:
    initial = urllib.request.Request(SOURCE_URL, headers={"User-Agent": USER_AGENT})
    response = urllib.request.urlopen(initial, timeout=60)
    content_type = response.headers.get_content_type()
    if content_type != "text/html":
        response.close()
        return initial
    page = response.read(128 * 1024).decode("utf-8", "replace")
    response.close()
    parser = FormParser()
    parser.feed(page)
    if not parser.action:
        raise RuntimeError("Google Drive did not return a recognized download/confirmation form")
    values = dict(parser.hidden)
    values.setdefault("id", FILE_ID)
    values.setdefault("export", "download")
    action = urllib.parse.urljoin("https://drive.google.com", parser.action)
    url = f"{action}?{urllib.parse.urlencode(values)}"
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def atomic_json(path: Path, payload: object) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = request_for_download()
    offset = partial.stat().st_size if partial.exists() else 0
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.headers.get_content_type() == "text/html":
            message = response.read(4096).decode("utf-8", "replace")
            raise RuntimeError(f"Expected archive, received HTML: {re.sub(r'<[^>]+>', ' ', message)[:500]}")
        append = offset > 0 and response.status == 206
        mode = "ab" if append else "wb"
        if offset and not append:
            offset = 0
        with partial.open(mode) as output:
            while block := response.read(8 * 1024 * 1024):
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if not zipfile.is_zipfile(partial):
            raise RuntimeError(f"Downloaded bytes are not a valid ZIP archive: {partial}")
    digest = sha256(partial)
    partial.replace(destination)
    manifest = {
        "source_url": SOURCE_URL,
        "google_drive_file_id": FILE_ID,
        "filename": destination.name,
        "size_bytes": destination.stat().st_size,
        "sha256": digest,
        "archive_valid_zip": True,
        "license": "verify from official dataset materials; repository code license is not assumed to cover the data",
    }
    atomic_json(destination.with_suffix(destination.suffix + ".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    download(Path("content/datasets/xlcost/raw/XLCoST_data.zip"))

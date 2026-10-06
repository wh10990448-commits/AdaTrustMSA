"""Download a large file with resume support and verify its SHA-256 digest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time
import urllib.request


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".part")
    downloaded = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(args.url)
    if downloaded:
        request.add_header("Range", f"bytes={downloaded}-")

    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=120) as response:
        if downloaded and response.status != 206:
            downloaded = 0
            partial.unlink(missing_ok=True)
        total = response.headers.get("Content-Length")
        total = downloaded + int(total) if total is not None else None
        mode = "ab" if downloaded else "wb"
        next_report = downloaded + 256 * 1024 * 1024
        with partial.open(mode) as output:
            while True:
                block = response.read(16 * 1024 * 1024)
                if not block:
                    break
                output.write(block)
                downloaded += len(block)
                if downloaded >= next_report:
                    elapsed = max(time.monotonic() - started, 1e-6)
                    progress = f"/{total}" if total is not None else ""
                    print(
                        f"downloaded={downloaded}{progress} "
                        f"rate_mib_s={downloaded / elapsed / 1024 / 1024:.1f}",
                        flush=True,
                    )
                    next_report += 256 * 1024 * 1024

    actual = file_sha256(partial)
    if actual.lower() != args.sha256.lower():
        raise RuntimeError(f"SHA-256 mismatch: expected {args.sha256}, got {actual}")
    partial.replace(args.output)
    print(f"verified={args.output} sha256={actual} bytes={downloaded}", flush=True)


if __name__ == "__main__":
    main()

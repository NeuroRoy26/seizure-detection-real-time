import os
from typing import Optional, Tuple

import requests


def download_if_needed(
    *,
    url: str,
    local_path: str,
    timeout_s: float = 30.0,
    previous_fingerprint: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Download a remote file to local_path if it’s missing or changed.

    Returns (did_download, new_fingerprint).
    Fingerprint is best-effort from HTTP headers (ETag / Last-Modified / Content-Length).

    Works well with public Google Cloud Storage URLs like:
      https://storage.googleapis.com/<bucket>/<path>/latest.h5
    """
    if not url:
        return (False, previous_fingerprint)

    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

    new_fingerprint = None
    try:
        head = requests.head(url, timeout=timeout_s, allow_redirects=True)
        head.raise_for_status()
        etag = head.headers.get("ETag")
        last_modified = head.headers.get("Last-Modified")
        content_length = head.headers.get("Content-Length")
        new_fingerprint = etag or last_modified or content_length
    except Exception:
        new_fingerprint = None

    if os.path.exists(local_path) and previous_fingerprint and new_fingerprint and new_fingerprint == previous_fingerprint:
        return (False, previous_fingerprint)

    if os.path.exists(local_path) and previous_fingerprint and new_fingerprint is None:
        # No reliable fingerprint (or HEAD unsupported) — avoid re-downloading aggressively.
        return (False, previous_fingerprint)

    tmp_path = f"{local_path}.tmp"
    with requests.get(url, stream=True, timeout=timeout_s, allow_redirects=True) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        etag = r.headers.get("ETag")
        last_modified = r.headers.get("Last-Modified")
        content_length = r.headers.get("Content-Length")
        new_fingerprint = etag or last_modified or content_length or new_fingerprint

    os.replace(tmp_path, local_path)
    return (True, new_fingerprint)


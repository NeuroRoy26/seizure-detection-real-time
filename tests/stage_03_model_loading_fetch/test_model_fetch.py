from pathlib import Path

from model_fetch import download_if_needed


class FakeHeadResponse:
    def __init__(self, headers):
        self.headers = headers

    def raise_for_status(self):
        return None


class FakeGetResponse:
    def __init__(self, headers, chunks):
        self.headers = headers
        self._chunks = chunks

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        del chunk_size
        for chunk in self._chunks:
            yield chunk

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_download_skips_when_fingerprint_matches(monkeypatch, tmp_path):
    out_file = tmp_path / "model.onnx"
    out_file.write_bytes(b"existing")

    def _unexpected_get(*args, **kwargs):
        raise AssertionError("GET should not be called")

    monkeypatch.setattr(
        "model_fetch.requests.head",
        lambda *args, **kwargs: FakeHeadResponse({"ETag": "same-etag"}),
    )
    monkeypatch.setattr("model_fetch.requests.get", _unexpected_get)

    did_download, fingerprint = download_if_needed(
        url="https://example.test/model.onnx",
        local_path=str(out_file),
        previous_fingerprint="same-etag",
    )
    assert did_download is False
    assert fingerprint == "same-etag"
    assert out_file.read_bytes() == b"existing"


def test_download_writes_file_when_changed(monkeypatch, tmp_path):
    out_file = Path(tmp_path) / "model.onnx"

    monkeypatch.setattr(
        "model_fetch.requests.head",
        lambda *args, **kwargs: FakeHeadResponse({"ETag": "new-etag"}),
    )
    monkeypatch.setattr(
        "model_fetch.requests.get",
        lambda *args, **kwargs: FakeGetResponse({"ETag": "new-etag"}, [b"abc", b"123"]),
    )

    did_download, fingerprint = download_if_needed(
        url="https://example.test/model.onnx",
        local_path=str(out_file),
        previous_fingerprint="old-etag",
    )
    assert did_download is True
    assert fingerprint == "new-etag"
    assert out_file.read_bytes() == b"abc123"

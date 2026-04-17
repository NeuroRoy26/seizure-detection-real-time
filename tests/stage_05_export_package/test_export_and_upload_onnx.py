import sys

import pytest

import export_and_upload_onnx as exporter


def test_run_invokes_subprocess(monkeypatch):
    calls = []
    monkeypatch.setattr(exporter.subprocess, "check_call", lambda cmd: calls.append(cmd))
    exporter._run(["echo", "hello"])
    assert calls == [["echo", "hello"]]


def test_main_raises_when_pt_missing(monkeypatch, tmp_path):
    missing = tmp_path / "missing.pt"
    monkeypatch.setattr(sys, "argv", ["export_and_upload_onnx.py", "--pt", str(missing), "--onnx", str(tmp_path / "out.onnx")])
    with pytest.raises(SystemExit) as exc_info:
        exporter.main()
    assert "Missing .pt file" in str(exc_info.value)


def test_main_exports_and_uploads_when_bucket_provided(monkeypatch, tmp_path):
    pt_path = tmp_path / "model.pt"
    onnx_path = tmp_path / "latest.onnx"
    pt_path.write_bytes(b"weights")

    export_calls = []
    run_calls = []
    monkeypatch.setattr(exporter, "export_onnx", lambda pt, onnx: export_calls.append((pt, onnx)))
    monkeypatch.setattr(exporter, "_run", lambda cmd: run_calls.append(cmd))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_and_upload_onnx.py",
            "--pt",
            str(pt_path),
            "--onnx",
            str(onnx_path),
            "--bucket",
            "my-bucket",
            "--object",
            "models/latest.onnx",
        ],
    )

    exporter.main()

    assert len(export_calls) == 1
    assert run_calls == [["gcloud", "storage", "cp", str(onnx_path.resolve()), "gs://my-bucket/models/latest.onnx"]]

"""
Local helper to export your trained EEGNet1D PyTorch weights (.pt) to ONNX and (optionally) upload to GCS.

This mirrors the last “export + upload” steps from `Seizure_Detection.ipynb`, but without Colab/Drive paths.

Usage examples:
  python3 export_and_upload_onnx.py --pt GLOBAL_eeg_model_TOP10.pt --onnx latest.onnx
  python3 export_and_upload_onnx.py --pt GLOBAL_eeg_model_TOP10.pt --onnx latest.onnx --bucket seizurebucket --object models/latest.onnx --make-public

Notes:
  - Export requires PyTorch installed locally.
  - Upload requires `gcloud` CLI authenticated locally.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def export_onnx(pt_path: Path, onnx_path: Path) -> None:
    import torch

    class EEGNet1D(torch.nn.Module):
        def __init__(self, num_channels: int = 10):
            super().__init__()
            self.conv1 = torch.nn.Conv1d(in_channels=num_channels, out_channels=16, kernel_size=5, padding=2)
            self.relu1 = torch.nn.ReLU()
            self.pool1 = torch.nn.MaxPool1d(kernel_size=2)

            self.conv2 = torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
            self.relu2 = torch.nn.ReLU()
            self.pool2 = torch.nn.MaxPool1d(kernel_size=2)

            self.flatten = torch.nn.Flatten()
            self.fc1 = torch.nn.Linear(32 * 64, 64)
            self.relu3 = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(64, 2)

        def forward(self, x):
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.flatten(x)
            x = self.relu3(self.fc1(x))
            x = self.fc2(x)
            return x

    state = torch.load(str(pt_path), map_location="cpu")
    model = EEGNet1D(num_channels=10)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.zeros((1, 10, 256), dtype=torch.float32)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["eeg"],
        output_names=["logits"],
        opset_version=17,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pt", required=True, help="Path to GLOBAL_eeg_model_TOP10.pt (state_dict)")
    p.add_argument("--onnx", required=True, help="Output ONNX path, e.g. latest.onnx")
    p.add_argument("--bucket", default="", help="GCS bucket name, e.g. seizurebucket")
    p.add_argument("--object", dest="object_path", default="models/latest.onnx", help="GCS object path")
    p.add_argument("--make-public", action="store_true", help="Attempt to make object publicly readable")
    args = p.parse_args()

    pt_path = Path(args.pt).expanduser().resolve()
    onnx_path = Path(args.onnx).expanduser().resolve()

    if not pt_path.exists():
        raise SystemExit(f"Missing .pt file: {pt_path}")

    print(f"Exporting ONNX: {pt_path} -> {onnx_path}")
    export_onnx(pt_path, onnx_path)
    print(f"✅ Wrote: {onnx_path}")

    if args.bucket:
        gcs_uri = f"gs://{args.bucket}/{args.object_path}"
        _run(["gcloud", "storage", "cp", str(onnx_path), gcs_uri])
        public_url = f"https://storage.googleapis.com/{args.bucket}/{args.object_path}"
        print(f"✅ Uploaded: {gcs_uri}")
        print(f"Public URL (MODEL_URL): {public_url}")

        if args.make_public:
            # This may fail if Public Access Prevention is enabled; bucket policy must allow it.
            try:
                _run(["gcloud", "storage", "objects", "update", gcs_uri, "--add-acl-grant=entity=AllUsers,role=READER"])
            except Exception:
                print("⚠️ Could not update object ACL (bucket may have Public Access Prevention enabled).")


if __name__ == "__main__":
    main()


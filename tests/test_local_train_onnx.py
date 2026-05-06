import os
import tempfile

import pytest

import numpy as np

h5py = pytest.importorskip("h5py")


def _make_h5(n=100, channels=10, samples=256):
    tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    tmp.close()
    with h5py.File(tmp.name, "w") as f:
        f.create_dataset(
            "X", data=np.random.randn(n, channels, samples).astype("float32")
        )
        f.create_dataset("y", data=np.random.randint(0, 2, n).astype("int32"))
    return tmp.name


def test_hdf5_generator_len():
    from local_train_onnx import HDF5Generator

    path = _make_h5(n=100)
    gen = HDF5Generator(path, np.arange(100), batch_size=32)
    assert len(gen) == 4  # ceil(100/32)
    os.unlink(path)


def test_hdf5_generator_getitem_shapes():
    from local_train_onnx import HDF5Generator

    path = _make_h5(n=50, channels=10, samples=256)
    gen = HDF5Generator(path, np.arange(50), batch_size=16)
    X_batch, y_batch = gen[0]
    assert X_batch.shape == (16, 10, 256)
    assert y_batch.shape == (16,)
    os.unlink(path)


def test_build_api_compliant_cnn_output_shape():
    from local_train_onnx import build_api_compliant_cnn

    model = build_api_compliant_cnn(input_shape=(10, 256))
    assert model.output_shape == (None, 2)  # 2 logits


def test_build_api_compliant_cnn_has_batch_norm():
    from local_train_onnx import build_api_compliant_cnn

    model = build_api_compliant_cnn(input_shape=(10, 256))
    layer_types = [type(layer).__name__ for layer in model.layers]
    assert "BatchNormalization" in layer_types


def test_print_metrics_perfect_predictions(capsys):
    from local_train_onnx import print_metrics

    y = np.array([0, 1, 0, 1])
    print_metrics("TEST", y, y)
    out = capsys.readouterr().out
    assert "Accuracy  : 1.0000" in out
    assert "F1-Score  : 1.0000" in out


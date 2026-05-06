import os
import tempfile

import pytest

import numpy as np
from unittest.mock import MagicMock

# h5py must actually be present (added to requirements-dev.txt)
h5py = pytest.importorskip("h5py", reason="h5py not installed")


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
    try:
        gen = HDF5Generator(path, np.arange(100), batch_size=32)
        assert len(gen) == 4  # ceil(100/32)
    finally:
        os.unlink(path)


def test_hdf5_generator_getitem_shapes():
    from local_train_onnx import HDF5Generator

    path = _make_h5(n=50, channels=10, samples=256)
    try:
        gen = HDF5Generator(path, np.arange(50), batch_size=16)
        X_batch, y_batch = gen[0]
        assert X_batch.shape == (16, 10, 256)
        assert y_batch.shape == (16,)
    finally:
        os.unlink(path)


def _install_tf_keras_fakes(monkeypatch):
    """
    Installs minimal fakes into local_train_onnx so we can unit-test the model
    wiring in CI without requiring TensorFlow to be installed.
    """
    import local_train_onnx as m

    call_trace = []

    def _make_layer_factory(layer_type: str):
        def _factory(*args, **kwargs):
            layer = MagicMock(name=f"{layer_type}Layer")
            layer._layer_type = layer_type
            layer._init_args = args
            layer._init_kwargs = kwargs

            def _apply(x, *c_args, **c_kwargs):
                call_trace.append(
                    {
                        "layer_type": layer_type,
                        "init_args": args,
                        "init_kwargs": kwargs,
                        "call_args": (x,) + c_args,
                        "call_kwargs": c_kwargs,
                    }
                )
                return f"{layer_type}_OUT"

            layer.side_effect = _apply
            return layer

        return _factory

    # Fake layers module
    fake_layers = MagicMock(name="fake_layers")
    fake_layers.Input = MagicMock(name="Input", side_effect=lambda *a, **k: "INPUT_TENSOR")
    fake_layers.Reshape = _make_layer_factory("Reshape")
    fake_layers.Conv2D = _make_layer_factory("Conv2D")
    fake_layers.BatchNormalization = _make_layer_factory("BatchNormalization")
    fake_layers.Resizing = _make_layer_factory("Resizing")
    fake_layers.GlobalAveragePooling2D = _make_layer_factory("GlobalAveragePooling2D")
    fake_layers.Dropout = _make_layer_factory("Dropout")
    fake_layers.Dense = _make_layer_factory("Dense")

    # Fake base model from tf.keras.applications
    base = MagicMock(name="MobileNetV2Base")
    base.trainable = True

    def _base_call(x, *args, **kwargs):
        call_trace.append(
            {
                "layer_type": "BaseModel",
                "init_args": (),
                "init_kwargs": {},
                "call_args": (x,) + args,
                "call_kwargs": kwargs,
            }
        )
        return "BASE_OUT"

    base.side_effect = _base_call

    fake_apps = MagicMock(name="fake_applications")
    fake_apps.MobileNetV2 = MagicMock(name="MobileNetV2", return_value=base)

    fake_keras = MagicMock(name="fake_keras")
    fake_keras.applications = fake_apps

    fake_tf = MagicMock(name="fake_tf")
    fake_tf.keras = fake_keras

    # Fake models.Model
    fake_models = MagicMock(name="fake_models")
    fake_models.Model = MagicMock(
        name="Model",
        side_effect=lambda *a, **k: MagicMock(name="KerasModel"),
    )

    monkeypatch.setattr(m, "tf", fake_tf, raising=True)
    monkeypatch.setattr(m, "layers", fake_layers, raising=True)
    monkeypatch.setattr(m, "models", fake_models, raising=True)

    return m, call_trace, base, fake_apps


def test_build_api_compliant_model_has_required_bridge_and_frozen_backbone(monkeypatch):
    m, call_trace, base, fake_apps = _install_tf_keras_fakes(monkeypatch)

    _ = m.build_api_compliant_cnn(input_shape=(10, 256))

    # Input contract preserved: (10, 256)
    m.layers.Input.assert_called_once()
    assert m.layers.Input.call_args.kwargs.get("shape") == (10, 256)

    # Immediately after input: reshape to (10, 256, 1)
    reshape_inits = [
        t for t in call_trace if t["layer_type"] == "Reshape"
    ]
    assert reshape_inits, "Expected a Reshape layer to be applied"
    first_reshape = reshape_inits[0]
    assert first_reshape["init_args"][0] == (10, 256, 1)
    assert first_reshape["init_kwargs"].get("name") == "Bridge_Reshape_10x256x1"

    # Expand to 3 channels for ImageNet backbone via 1x1 Conv2D (filters=3)
    conv2d_events = [t for t in call_trace if t["layer_type"] == "Conv2D"]
    assert conv2d_events, "Expected a Conv2D bridge to 3 channels"
    bridge_conv2d = conv2d_events[0]
    assert bridge_conv2d["init_kwargs"].get("filters") == 3
    assert bridge_conv2d["init_kwargs"].get("kernel_size") == (1, 1)
    assert bridge_conv2d["init_kwargs"].get("padding") == "same"
    assert bridge_conv2d["init_kwargs"].get("name") == "Bridge_ToRGB_1x1Conv"

    # Base model must be ImageNet MobileNetV2 include_top=False
    fake_apps.MobileNetV2.assert_called_once()
    assert fake_apps.MobileNetV2.call_args.kwargs["include_top"] is False
    assert fake_apps.MobileNetV2.call_args.kwargs["weights"] == "imagenet"
    assert fake_apps.MobileNetV2.call_args.kwargs["input_shape"] == (224, 224, 3)

    # Base model must be frozen
    assert base.trainable is False

    # Base model forward call must use training=False
    base_calls = [t for t in call_trace if t["layer_type"] == "BaseModel"]
    assert base_calls, "Expected base model to be called"
    assert base_calls[0]["call_kwargs"].get("training") is False


def test_build_api_compliant_model_outputs_logits_not_softmax(monkeypatch):
    m, call_trace, _base, _fake_apps = _install_tf_keras_fakes(monkeypatch)

    _ = m.build_api_compliant_cnn(input_shape=(10, 256))

    dense_events = [t for t in call_trace if t["layer_type"] == "Dense"]
    assert dense_events, "Expected a final Dense layer"
    last_dense = dense_events[-1]
    assert last_dense["init_args"][0] == 2
    assert last_dense["init_kwargs"].get("activation") == "linear"
    assert last_dense["init_kwargs"].get("name") == "PyTorch_Logit_Mimic"


def test_print_metrics_perfect_predictions(capsys):
    from local_train_onnx import print_metrics

    y = np.array([0, 1, 0, 1])
    print_metrics("TEST", y, y)
    out = capsys.readouterr().out
    assert "Accuracy  : 1.0000" in out
    assert "F1-Score  : 1.0000" in out


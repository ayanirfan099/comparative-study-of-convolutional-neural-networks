import subprocess
import sys

import numpy as np
import pytest
import torch

from vision_study.cli import run
from vision_study.core import build_model, gradcam, load_checkpoint, loaders, seed_all


def test_models_and_gradient():
    seed_all(4)
    m = build_model()
    x = torch.randn(2, 3, 32, 32)
    assert m(x).shape == (2, 10)
    cam = gradcam(m, x[:1], m.features[3])
    assert cam.shape == (32, 32) and np.isfinite(cam).all()
    with pytest.raises(ValueError):
        build_model("resnet18", freeze=True)
    assert build_model("resnet18")(x).shape == (2, 10)


def test_splits():
    (train, val, test), ids = loaders(".", limit=30, synthetic=True)
    assert not set(ids["train"]) & set(ids["validation"])
    assert next(iter(train))[0].shape[1:] == (3, 32, 32)
    assert len(test.dataset) > 0 and len(val.dataset) > 0
    with pytest.raises(ValueError):
        loaders(".", limit=0)


def test_end_to_end(tmp_path):
    a = run(output=str(tmp_path / "a"), epochs=1, limit=30, synthetic=True)
    b = run(output=str(tmp_path / "b"), epochs=1, limit=30, synthetic=True)
    assert a["test"] == b["test"]
    assert a["history"] == b["history"]
    m, _ = load_checkpoint(tmp_path / "a/best.pt")
    assert m(torch.zeros(1, 3, 32, 32)).shape == (1, 10)
    assert (tmp_path / "a/gradcam.png").exists()


def test_cli():
    assert (
        subprocess.run(
            [sys.executable, "-m", "vision_study.cli", "--help"], check=False, capture_output=True
        ).returncode
        == 0
    )

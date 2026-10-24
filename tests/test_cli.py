import tempfile
from pathlib import Path

import numpy as np
import pytest

from cnn_lstm_bp.cli import predict


def test_cli():
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.npy"
        np.save(bad, np.zeros((10, 1), np.float32))
        with pytest.raises(SystemExit):
            predict(bad, "sbp", Path(tmp) / "missing.pth")

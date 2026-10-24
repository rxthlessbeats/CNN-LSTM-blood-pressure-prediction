import numpy as np

from cnn_lstm_bp.data import resample_to


def test_data():
    original = np.arange(625, dtype=np.float32)
    assert np.array_equal(resample_to(original, 625), original)
    assert len(resample_to(original, 100)) == 100
    assert len(resample_to(np.arange(10, dtype=np.float32), 625)) == 625

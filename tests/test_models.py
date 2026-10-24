import torch

from cnn_lstm_bp.models import BP_Model, BP_Model_DO


def test_models():
    batch = torch.zeros(2, 64, 2)
    for cls in (BP_Model_DO, BP_Model):
        out = cls().eval()(batch)
        assert out.shape == (2, 1)
        assert torch.isfinite(out).all()

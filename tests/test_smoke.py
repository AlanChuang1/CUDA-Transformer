import torch
import pytest
import ext as ops

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_extension_smoke():
    x = torch.randn(2, 3, 4096, device="cuda", dtype=torch.float32).contiguous()
    w = torch.randn(4096, device="cuda", dtype=torch.float32).contiguous()
    y = ops.rmsnorm_fwd(x, w, 1e-6)
    assert y.shape == x.shape

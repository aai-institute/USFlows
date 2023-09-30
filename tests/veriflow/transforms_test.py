import torch

from src.veriflow.transforms import ScaleTransform

def test_scale_transform():
    # Test input
    dim = 10
    transform = ScaleTransform(10)
    with torch.no_grad():
        transform.scale.copy_(torch.ones(dim) * 2)
    x = torch.ones(dim)
    
    # Test forward, inverse, and log det
    y = transform(x)
    assert (y == 2 * x).all()
    assert (transform.backward(y) == x).all()
    log_det = transform.log_abs_det_jacobian(x, y)
    assert log_det == dim * torch.log(torch.tensor(2))
    
import numpy as np
import torch

# debug utils
def TORCH_CHECK_NAN(x: torch.Tensor) -> None:
    """Checks for NaNs in the input tensor."""
    assert torch.isnan(x).sum() == 0


def TORCH_CHECK_FINITE(x: torch.Tensor) -> None:
    """Checks if all entries in the input tensor are finite."""
    assert torch.isfinite(x).all()


def NUMPY_CHECK_NAN(x: np.ndarray) -> None:
    """Checks for NaNs in the input array."""
    assert np.isnan(x).sum() == 0


def NUMPY_CHECK_FINITE(x: np.ndarray) -> None:
    """Checks if all entries in the input array are finite."""
    assert np.isfinite(x).all()

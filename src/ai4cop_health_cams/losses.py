import torch


def energy_score_loss(y_pred: torch.Tensor, y_obs: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Calculates the energy score loss (vectorized version)
    Args:
        y_pred: forecast realizations, shape (bs, nvar, lat, lon, members)
        y_true: ground truth ("observations"), shape (bs, nvar, lat, lon)
        beta: beta exponent for the energy loss (beta = 1.0 results in the CRPS of the ensemble distribution)
    Returns:
        The energy score loss.
    """
    bs = y_pred.shape[0]
    m = y_pred.shape[-1]

    tmp_a = (2.0 / m) * torch.mean(
        torch.sum(
            torch.pow(
                torch.linalg.norm((y_pred - y_obs[..., None]).reshape(bs, -1, m), dim=1, ord=2),
                beta,
            ),
            dim=-1,
        )
    )

    # this is will copy some data so it's (probably) not super-efficient
    y_pred = y_pred.reshape(bs, -1, m).permute(0, 2, 1).contiguous()

    tmp_b = (
        1.0
        / (m * (m - 1))
        * torch.mean(
            torch.sum(
                torch.pow(torch.cdist(y_pred, y_pred, p=2), beta),
                axis=(1, 2),
            ),
            dim=0,
        )
    )

    return tmp_a - tmp_b

import torch
import math


def forward(feat, numangle, numrho):
    """
    feat: [N, C, H, W]
    output: [N, C, numangle, numrho]
    """
    N, C, H, W = feat.shape
    device = feat.device

    # kąty
    theta = torch.linspace(0, math.pi, numangle, device=device)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    # siatka współrzędnych
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )

    xs = xs - W // 2
    ys = ys - H // 2

    xs = xs.reshape(-1)
    ys = ys.reshape(-1)

    output = torch.zeros((N, C, numangle, numrho), device=feat.device, dtype=feat.dtype)

    # główna pętla po kątach
    for k in range(numangle):
        r = xs * cos_t[k] + ys * sin_t[k]
        r = torch.round(r).long() + numrho // 2

        valid = (r >= 0) & (r < numrho)
        r_valid = r[valid]

        for n in range(N):
            for c in range(C):
                vals = feat[n, c].reshape(-1)[valid]
                output[n, c, k].scatter_add_(0, r_valid, vals)

    return [output]  # żeby pasowało do oryginalnego API

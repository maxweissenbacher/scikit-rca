from collections import Counter

import numpy as np
import scipy.stats
import torch


def icc11(subjects, values, alpha=0.05, r0=0, return_stats=True):
    """Compute ICC(1,1).

    `subjects` should be a list of subject ids.  `values` should be a list of
    the same length of observations over which to take ICC.

    Requires that there be an equal number of observations fer subject.

    Converted from the matlab function:
    https://www.mathworks.com/matlabcentral/fileexchange/22099-intraclass-correlation-coefficient-icc

    """
    # Reformat from a 1D matrix of subjects and a 1D matrix of values to a 2D
    # matrix, subjects x repeats.  This is the vectorised way of doing:
    # data = [values[subjects==s] for s in list(sorted(set(subjs)))]
    subjs, indices = np.unique(subjects, return_inverse=True)
    data = np.asarray(
        np.split(
            np.asarray(values)[np.argsort(indices)],
            np.cumsum(np.bincount(indices))[:-1],
        )
    )
    assert data.ndim == 2
    n, k = data.shape
    # Perform ICC calculations
    msr = np.var(np.mean(data, axis=1), ddof=1) * k
    msw = np.mean(np.var(data, axis=1, ddof=1))
    r = (msr - msw) / (msr + (k - 1) * msw)
    if not return_stats:
        return r
    # Compute confidence interval and p-value
    F = (msr / msw) * (1 - r0) / (1 + (k - 1) * r0)
    df1 = n - 1
    df2 = n * (k - 1)
    p = 1 - scipy.stats.f.cdf(F, df1, df2)
    FL = (msr / msw) / scipy.stats.f.ppf(1 - alpha / 2, n - 1, n * (k - 1))
    FU = (msr / msw) * scipy.stats.f.ppf(1 - alpha / 2, n * (k - 1), n - 1)
    LB = (FL - 1) / (FL + (k - 1))
    UB = (FU - 1) / (FU + (k - 1))
    return r, (LB, UB), p


# Loss computations used in training
def compute_same_diff_from_label(label):
    subjnum = label[:, 0]
    scannum = label[:, 1]
    same = (subjnum[:, None] == subjnum[None, :]) & (scannum[:, None] != scannum[None, :])
    diff = subjnum[:, None] != subjnum[None, :]
    return same, diff


def contrastive_loss(embedding, label, eps):
    """
    Computes a max-margin contrastive loss objective for a set of embeddings.
    """
    same, diff = compute_same_diff_from_label(label)
    # Gives matrix with (i,j) = L2 norm of (output[i:] - output[j:])
    dist = torch.cdist(embedding, embedding, p=2)
    loss_same = torch.mean(torch.pow(torch.masked_select(dist, same), 2))
    loss_diff = torch.mean(torch.pow(torch.clamp(eps - torch.masked_select(dist, diff), 0), 2))
    return loss_same + loss_diff


def info_nce(embedding, label, temp=0.1):
    """
    Computes the InfoNCE loss objective for two sets of embeddings using the L2 norm
    instead of cosine similarity since we use one-dimensional embeddings most of the
    time.
    Input:
        embedding: shape [N, embedding_dim]
    """
    same, diff = compute_same_diff_from_label(label)
    # Gives matrix with (i,j) = L2 norm of (output[i:] - output[j:])
    dist = torch.cdist(embedding, embedding, p=2)
    # Convert distance to similarity
    sim = -dist / temp  # [N, N]
    # Sum distance over all positive pairs; minimizing this maximizes similarity.
    num = -torch.mean(torch.masked_select(sim, same))
    # Sum distance over all pairs; minimizing this maximizes similarity overall.
    denom = torch.mean(torch.logsumexp(sim, dim=-1))
    # Only keep positive log-probs
    loss = num + denom
    return loss

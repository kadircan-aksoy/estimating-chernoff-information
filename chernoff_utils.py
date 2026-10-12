'''
Dataset generators and utils for Chernoff Information estimation
'''

import numpy as np
import torch
from scipy.stats import truncnorm as sp_truncnorm, truncexpon as sp_truncexpon, norm
from scipy.optimize import brentq
from sklearn.neighbors import NearestNeighbors

def kth_dists(src, ref, k, self_lo=False):
    """
        Euclidean distance from each point in src to its k-th nearest neighbour in ref.
        src: array (n,d), query points
        ref: array (m,d), reference sample
        k: neighbour rank (1 = nearest)
        self_lo: leave-one-out, set when src is ref; skips each point's zero distance to itself
        Returns array (n,), floored at 1e-8 so that log distances stay finite.
    """
    nn = NearestNeighbors(
        n_neighbors=k + (1 if self_lo else 0),
        metric="minkowski",
        p=2
    )
    nn.fit(ref)
    dists, _ = nn.kneighbors(src, return_distance=True)

    # k-th (or (k+1)-th if self_lo) neighbor distances; floor to avoid zeros
    out = dists[:, k] if self_lo else dists[:, k - 1]

    return np.maximum(out, 1e-8)


def truncnorm(mu1, sigma1, mu2, sigma2, a=0.0, b=1.0, sample_size=1000, rng=None):
    def sample_truncnorm(n, mu, sigma, a=0.0, b=1.0, rng=None):
        rng = np.random.default_rng(rng)
        mu, sigma = np.atleast_1d(mu), np.atleast_1d(sigma)
        al, bl = (a - mu) / sigma, (b - mu) / sigma
        return sp_truncnorm.rvs(al, bl, loc=mu, scale=sigma, size=(n, len(mu)), random_state=rng)

    rng = np.random.default_rng(rng)
    X = sample_truncnorm(sample_size, mu1, sigma1, a, b, rng)
    Y = sample_truncnorm(sample_size, mu2, sigma2, a, b, rng)
    return torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(Y, dtype=torch.float32), 0, 0

def truncexp(lam1, lam2, b=1.0, sample_size=1000, rng=None):
    def sample_truncexp(n, lam, b=1.0, rng=None):
        rng = np.random.default_rng(rng)
        lam = np.atleast_1d(lam)
        return sp_truncexpon.rvs(b * lam, scale=1 / lam, size=(n, len(lam)), random_state=rng)
    rng = np.random.default_rng(rng)
    X = sample_truncexp(sample_size, lam1, b, rng)
    Y = sample_truncexp(sample_size, lam2, b, rng)
    l1, l2 = np.atleast_1d(lam1), np.atleast_1d(lam2)
    def logCs(s):
        ls = s * l1 + (1 - s) * l2
        Z1, Z2, Zs = 1 - np.exp(-l1 * b), 1 - np.exp(-l2 * b), 1 - np.exp(-ls * b)
        return np.sum(s * np.log(l1) + (1 - s) * np.log(l2) - np.log(ls)
                      + np.log(Zs) - s * np.log(Z1) - (1 - s) * np.log(Z2))
    h = 1e-6
    s_star = brentq(lambda s: (logCs(s + h) - logCs(s - h)) / (2 * h), h, 1 - h)
    C = -logCs(s_star)
    return torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(Y, dtype=torch.float32), s_star, C

def symmetric_truncnorm(sample_size=1000, delta=0.1, sigma=0.2, dim=1, rng=None):
    rng = np.random.default_rng(rng)
    mu1, mu2 = 0.5 - delta, 0.5 + delta
    X = sp_truncnorm.rvs(-mu1 / sigma, (1 - mu1) / sigma, loc=mu1, scale=sigma,
                         size=(sample_size, dim), random_state=rng)
    Y = sp_truncnorm.rvs(-mu2 / sigma, (1 - mu2) / sigma, loc=mu2, scale=sigma,
                         size=(sample_size, dim), random_state=rng)
    Z = norm.cdf((1 - mu1) / sigma) - norm.cdf(-mu1 / sigma)
    Zh = 2 * norm.cdf(0.5 / sigma) - 1
    true_c = dim * ((2 * delta) ** 2 / (8 * sigma ** 2) + np.log(Z) - np.log(Zh))
    return (torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(Y, dtype=torch.float32),
            0.5, true_c)

def gaussian_mean(m_p = 1, m_q = -1, sigma = 1, sample_size=1000):
    P = m_p + sigma*torch.randn(sample_size, 1)
    Q = m_q + sigma*torch.randn(sample_size, 1)

    # Formula taken from Nielsen 2022, simplest case
    true_c = ((m_p - m_q) ** 2) / 8*sigma
    true_s = 0.5
    return P, Q, true_s, true_c
    
def gaussian_variance(sig_p=1., sig_q=4., m=0., sample_size=1000):
    v_p, v_q = sig_p**2, sig_q**2
    P = torch.tensor(m + sig_p * np.random.randn(sample_size, 1), dtype=torch.float32)
    Q = torch.tensor(m + sig_q * np.random.randn(sample_size, 1), dtype=torch.float32)

    # Formula taken from Nielsen 2022, note the mean m is irrelevant as long as they're equal
    true_s = -(v_p * np.log(2*v_q) - v_q - v_p * np.log(2*v_p) + v_p) / ((v_q - v_p) * np.log(2*v_q) - np.log(2*v_p) * v_q + v_p * np.log(2*v_p))
    true_c = -((v_q - v_p) * np.log((v_q * np.log(2*v_q) - np.log(2*v_p) * v_q) / (v_q - v_p)) - v_q * np.log(2*v_q) + (np.log(2*v_p) + 1) * v_q - v_p) / (2*v_q - 2*v_p)
    return P, Q, true_s, true_c

def direct_gaussian(sample_size=1000):
    P = torch.tensor(np.random.randn(sample_size, 1), dtype=torch.float32)
    Q = torch.tensor(1.0 + 2.0 * np.random.randn(sample_size, 1), dtype=torch.float32)

    return P, Q, 0.4216, 0.1155

def bernoulli(p=0.3, q=0.7, sample_size=1000):

    P = torch.bernoulli(torch.full((sample_size, 1), p))
    Q = torch.bernoulli(torch.full((sample_size, 1), q))

    def lhs(s):
        return p**s * (1-p)**(1-s) / (q**s * (1-q)**(1-s)) - 1

    true_s = brentq(lhs, 1e-6, 1 - 1e-6)
    true_c = -np.log(p**true_s * q**(1-true_s) + (1-p)**true_s * (1-q)**(1-true_s))

    return P, Q, true_s, true_c


DATASETS = {
    "truncnorm": truncnorm,
    "truncexp": truncexp,
    "symmetric_truncnorm": symmetric_truncnorm,
    "gaussian_mean": gaussian_mean,
    "gaussian_variance": gaussian_variance,
    "direct_gaussian": direct_gaussian,
    "bernoulli": bernoulli,
}


def get_dataset(name=None, **kwargs):
    '''
    Dispatch to a dataset generator by keyword, e.g. get_dataset("bernoulli", p=0.3, q=0.7).
    With no name given, defaults to "symmetric_pair".
    '''
    if name is None:
        return symmetric_truncnorm(**kwargs)
    try:
        dataset_fn = DATASETS[name]
    except KeyError:
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {sorted(DATASETS)}")
    return dataset_fn(**kwargs)
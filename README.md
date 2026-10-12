# On the Estimation of Chernoff Information

Code for the paper [*On the Estimation of Chernoff Information*](https://arxiv.org/abs/2608.17916) (Aksoy & Jung, 2026).

We estimate the Chernoff information between two distributions from samples. A k-nearest-neighbour estimator of the derivative of the Chernoff coefficient is used with bisection to find the optimal exponent $s^*$.

## Background

Chernoff information between distributions $P$ and $Q$ is the maximum (over $s \in (0,1)$) of their (un-normalized) Renyi-divergence:

$$
C(P,Q) = \max_{s \in (0,1)} -\log \int p(x)^s \, q(x)^{1-s} \, dx.
$$

It is the optimal error exponent in Bayesian binary hypothesis testing: with $N$ i.i.d. observations, the minimum probability of error decays as $e^{-N\,C(P,Q)}$. Since $C_s$ is concave in $s$, the maximiser $s^*$ is unique and can be found by bisection on the sign of the derivative.

## Estimators

Let $x_1,\dots,x_n \sim P$ and $y_1,\dots,y_m \sim Q$ in $\mathbb{R}^d$. For each $x_i$, let $\rho_k(i)$ be the distance to its $k$-th nearest neighbour among the other $P$ samples, and $\nu_k(i)$ the distance to its $k$-th nearest neighbour among the $Q$ samples. The ratio

$$
r_i = \frac{(n-1)\,\rho_k(i)^d}{m\,\nu_k(i)^d}
$$

is a kNN estimate of $q(x_i)/p(x_i)$.

**Chernoff $s$-divergence** (`c_s_hat`):

$$
\hat C_s = -\log \left( B_{k,s}\,\frac{1}{n}\sum_{i=1}^n r_i^{\,1-s} \right),
\qquad
B_{k,s} = \frac{\Gamma(k)^2}{\Gamma(k-s+1)\,\Gamma(k+s-1)},
$$

where $B_{k,s}$ corrects the bias of the kNN ratio.

**Derivative estimator** (`c_s_prime_hat`) estimates $D_s = -\frac{d}{ds}C_s$:

$$
\hat D_s = \psi(k-s+1) - \psi(k+s-1) - \frac{\sum_{i=1}^n r_i^{\,1-s}\,\log r_i}{\sum_{i=1}^n r_i^{\,1-s}},
$$

where $\psi$ is the digamma function. The first two terms come from differentiating $\log B_{k,s}$, and the last term is a weighted average of $\log r_i$ with weights $r_i^{1-s}$.

**Bisection** (`nielsen_bisect_direct_derivative`): start from $[a,b] = [0,1]$, evaluate $\hat D_s$ at the midpoint $s$, and apply bisection on the interval using the sign of $\hat D_s$. After the final iteration, return $\hat s^*$ and $\hat C = \hat C_{\hat s^*}$.


## Contents

The notebook includes the following experiments: 

1) Synthetic datasets from the paper as sanity checks.
2) MNIST experiment estimating the Chernoff Info between different digits (requires separate MNIST download).
3) A dimension scaling experiment to gauge sample complexity.

## How to run

```bash
pip install -r ChernoffInformation/requirements.txt
jupyter lab
```

## Citation

If you use this code, please cite:

```bibtex
@misc{aksoy2026chernoff,
  title         = {On the Estimation of Chernoff Information},
  author        = {Aksoy, Kadircan and Jung, Peter},
  year          = {2026},
  eprint        = {2608.17916},
  archivePrefix = {arXiv},
  primaryClass  = {cs.IT},
  doi           = {10.48550/arXiv.2608.17916},
  url           = {https://arxiv.org/abs/2608.17916}
}
```

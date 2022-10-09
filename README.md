# BQNeRF

### What happens if we replace the rendering function of a NeRF with Bayesian quadrature?


This repo explores what happens if we assume transparency along the ray can be modelled using a Matern Kernel, and using this to compute the expected colour along each ray from samples drawn along these, along with it's uncertainty. This enables the use of a Gaussian log likelihood rendering loss, which accounts for sampling that may not necessarily be informative. This repository implements a very basic coarse NeRF, and is based heavily on this excellent [NeRF from nothing tutorial](https://towardsdatascience.com/its-nerf-from-nothing-build-a-vanilla-nerf-with-pytorch-7846e4c45666). Let's start with some background.

### Bayesian quadrature

Bayesian quadrature typically uses a Gaussian process (GP) to apprximate the underlying integrand $f(x)$ using a set of samples $\[x_i, f(x_i)\]$, and then computes the integral by integrating the GP. Let's assume we model the integrand using a GP with mean function $m(x)$ and covariance function $k(x,x')$. The posterior distribution on $f$ is also a GP with mean and covariance:

$$ \mu(x) = m(x) + k(x,X)k(X,X)^{-1}f(X) $$

$$ \Sigma(x) = k(x,y) - k(x,X)k(X,X)^{-1}k(X,y) $$ 

Integrating this GP produces a distribution on the integral $v = \int f(x) dx$:

$$ \mathbb{E} \[{v(f)} \] = v[m] + v[k(.,X)]k(X,X)^{-1}f(X) $$

$$ \mathbb{V} \[{v(f)} \] = vv[k] + v[k(.,X)]k(X,X)^{-1}v[k(X,.)] $$

where

$$ v([k(.,x)] = \int k(x,y) dy $$

$$ vv([k(.,x)] = \int k(x,y)v(dx)v(dy) $$

For many kernel or covariance functions (eg. RBF, Matern, etc.), these can be computed analytically.

### Volumetric Rendering in NeRFs

The standard volumetric rendering equation using in NeRFs approximates the colour perceived at the end of a ray using Gaussian quadrature:




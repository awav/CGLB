# CGLB

Conjugate Gradient Lower Bound


## Installation

### CGLB

The repo has two version of CGLB model:

* TensorFlow: you need to install only [GPflow](https://github.com/GPflow/GPflow#installation) and its dependencies.
* PyTorch: you should install [GPytorch](https://github.com/cornellius-gp/gpytorch#installation) and its dependencies.

### CGLB experiments

We base our command line interface on [click](https://pypi.org/project/click), and with [xpert](https://github.com/awav/xpert) manage multiple experiment runs on different GPUs.

You can find the full list of requirements at `requirements.txt`.


## Cite

```
@inproceedings{artemevburt21tighter,
  title = {Tighter Bounds on the Log Marginal Likelihood of Gaussian Process Regression Using Conjugate Gradients},
  author = {Artemev, Artem and Burt, David R. and van der Wilk, Mark},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
  pages = {362--372},
  year = {2021},
}
```
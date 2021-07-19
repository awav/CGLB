# CGLB

Conjugate Gradient Lower Bound


## Installation

### CGLB

The repo has two version of CGLB model:

* TensorFlow: you need to install only [GPflow](https://github.com/GPflow/GPflow#installation) and its dependencies.
* PyTorch: you should install [GPytorch](https://github.com/cornellius-gp/gpytorch#installation) and its dependencies.

### CGLB experiments

The command line interface is based on [click](https://pypi.org/project/click), and with [xpert](https://github.com/awav/xpert) experiment manager you can run and organize many experiments on different GPUs (or CPU).

You can find the full list of requirements at `requirements.txt`.


Install (develop):

```
$ pip install -r requirements.txt
$ pip install -e .
```


### Run experiments

with CLI:

```
$ python cli.py --keops -b torch -l "./logs" -s 0 -t fp64 train -n 2000 -d snelson1d cglb -k Matern32 -m cglb -i ConditionalVariance -M 1024
```

with `xpert`:

```
$ xpert xpert-main.toml
```


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
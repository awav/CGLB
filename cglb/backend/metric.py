from typing import Callable, Dict, Union, Tuple
import torch
import tensorflow as tf
import numpy as np

Tensor = Union[tf.Tensor, torch.Tensor]
PairTensor = Tuple[Tensor, Tensor]


def call_metric_fns(
    *fns: Callable[[], Dict[str, Union[Tensor, np.ndarray, float]]]
) -> Dict[str, float]:
    results = [fn() for fn in fns]
    np_results = dict()
    for res in results:
        np_res = {k: float(np.array(v)) for k, v in res.items()}
        np_results.update(np_res)
    return np_results


def rmse_and_lpd_fn(
    error_logdensity_cb: Callable[[], Tuple[PairTensor, PairTensor]],
) -> Callable[[], Dict[str, float]]:
    def inner_func() -> Dict[str, float]:
        errs, logdens = error_logdensity_cb()
        errs = [np.array(e) for e in errs]
        logdens = [np.array(l) for l in logdens]

        train_errors, test_errors = errs
        train_lds, test_lds = logdens

        metrics = {
            "train/rmse": np.sqrt(np.mean(train_errors ** 2)),
            "test/rmse": np.sqrt(np.mean(test_errors ** 2)),
            "train/nlpd": -np.mean(train_lds),
            "test/nlpd": -np.mean(test_lds),
        }

        return {k: float(v) for k, v in metrics.items()}

    return inner_func
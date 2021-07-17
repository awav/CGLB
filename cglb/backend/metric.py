# Copyright 2021 The CGLB Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
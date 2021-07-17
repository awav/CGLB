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

import time
import numpy as np
from typing import Callable, Dict
from contextlib import contextmanager

import torch

from gpflow.monitor import MonitorTaskGroup, ToTensorBoard

import tensorflow as tf


class StopWatch:
    def __init__(self):
        self._start_time = None
        self._pause_time = None
        self._total_paused_time = None

    def started(self) -> bool:
        return self._start_time is not None

    def start(self):
        self._start_time = time.time()
        self._total_paused_time = 0.0

    def pause(self):
        self._pause_time = time.time()

    def resume(self):
        paused_time = time.time() - self._pause_time
        self._total_paused_time += paused_time
        self._pause_time = None

    def reset(self):
        self._start_time = None
        self._pause_time = None
        self._total_paused_time = None

    def get_elapsed_time(self):
        end_time = time.time()
        total_time = end_time - self._start_time
        total_elapsed_time = total_time - self._total_paused_time
        return total_elapsed_time

    def stop(self):
        total_elapsed_time = self.get_elapsed_time()
        self.reset()
        return total_elapsed_time


class ScalarsToTensorBoard(ToTensorBoard):
    def __init__(self, log_dir: str, callback: Callable[[], Dict[str, float]]):
        super().__init__(log_dir)
        self.callback = callback

    def run(self, **kwargs):
        values_map = self.callback(**kwargs)
        for name, value in values_map.items():
            tf.summary.scalar(name, value, step=self.current_step)


class Logger:
    def __init__(
        self,
        logdir: str,
        metrics_fn: Callable,
        model_parameters_fn: Callable,
        holdout_interval: int = 10,
        include_feval_log: bool = False,  # Control logging of CG steps each f evaluation
    ):
        self.holdout_interval = holdout_interval
        self.logdir = logdir
        self._metrics_fn = metrics_fn
        self._model_parameters_fn = model_parameters_fn
        self._logs = {}
        self.counter = 0
        self.include_feval_log = include_feval_log

        timer = StopWatch()
        self.timer = timer

    @property
    def logs(self) -> Dict:
        return self._logs

    def model_parameters_fn(self) -> Dict[str, np.ndarray]:
        def include(key) -> bool:
            return "inducing_point" not in key

        params = self._model_parameters_fn()
        return {k: v for k, v in params.items() if include(k)}

    def metrics_fn(self) -> Dict[str, np.ndarray]:
        def include(key) -> bool:
            prefixes = ["train", "test", "cg/", "loss"]
            return any([key.startswith(prefix) for prefix in prefixes])

        metrics = self._metrics_fn()
        return {k: v for k, v in metrics.items() if include(k)}

    def log(self, **kwargs):
        for k, v in kwargs.items():
            if k in self._logs:
                self._logs[k].append(v)
            else:
                self._logs[k] = [v]

    def log_for_feval(self, **kwargs):
        if self.include_feval_log:
            logs = {f"{k}-per-feval": v for k, v in kwargs.items()}
            self.log(**logs)

    @contextmanager
    def no_recording(self):
        holdout_interval = self.holdout_interval
        include_feval_log = self.include_feval_log
        self.holdout_interval = -1
        self.include_feval_log = False
        try:
            yield
        finally:
            self.holdout_interval = holdout_interval
            self.include_feval_log = include_feval_log

    def __call__(self, step, *args):
        # if (self.counter <= 10) or (self.counter % self.holdout_interval) == 0:

        iteration = self.counter
        self.counter += 1

        if self.holdout_interval < 0:
            return

        if (iteration % self.holdout_interval) == 0:

            elapsed_time = self.timer.get_elapsed_time()
            self.timer.pause()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            params = self.model_parameters_fn()
            metrics = self.metrics_fn()

            tb_parametets = tb_format_parameters(params)
            tb_all_records = {"elapsed_time": elapsed_time, **tb_parametets, **metrics}

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            monitor = MonitorTaskGroup([ScalarsToTensorBoard(self.logdir, lambda: tb_all_records)])
            monitor(iteration)

            loss_value = metrics["loss"]
            print(f"{iteration} - loss={loss_value:.4f}", flush=True)

            self.log(
                iteration=iteration,
                elapsed_time=elapsed_time,
                params=params,
                **metrics,
            )

            self.timer.resume()


def tb_format_parameters(parameters: Dict) -> Dict:
    out_dict = {}
    monitor_keys = ["kernel", "likelihood"]
    for key, parameter in parameters.items():
        name = key.lstrip(".")
        if name.split(".")[0] not in monitor_keys:
            continue
        p = np.array(parameter).reshape(-1).squeeze()
        n = name.replace(".", "/", 1)
        if p.ndim == 0:
            out_dict[n] = p
        else:
            size = p.shape[0]
            for i in range(size):
                out_dict[f"{n}[{i}]"] = p[i]
    return out_dict
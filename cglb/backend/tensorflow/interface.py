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


from typing import Any, Tuple, Union, Optional, Callable, Dict
from functools import singledispatch
import warnings
import gpflow
import numpy as np
from pathlib import Path
import json_tricks
import os
import tensorflow as tf
from gpflow.utilities import parameter_dict, multiple_assign, positive
from gpflow import Parameter
from ..callbacks import Logger
from gpflow.models import GPR, SGPR
from .. import metric
from .models import CGLB, CGLBN2M, CGLBNM2, SGPRN2M
from ..config import (
    CGLBNM2Config,
    KernelConfig,
    SGPRN2MConfig,
    SquaredExponentialConfig,
    Matern32Config,
)
from ..config import (
    ModelConfig,
    GPRConfig,
    SGPRConfig,
    CGLBConfig,
    CGLBN2MConfig,
)


__all__ = [
    "set_default_jitter",
    "set_default_float",
    "get_default_float",
    "create_kernel",
    "create_model",
    "optimize",
    "save",
    "load",
    "metrics_fn",
    "model_parameters",
]


# Stubs

Tensor = tf.Tensor
PairTensor = Tuple[Tensor, Tensor]
Data = PairTensor


def jit_compile() -> bool:
    # WARN: change this line to turn on/off JIT for TF
    # compile = False
    compile = True

    if compile:
        print("JIT optimization applied")
    else:
        print("NO JIT optimization applied")

    return compile


def configure_backend(**kwargs):
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def set_default_jitter(value: float) -> None:
    gpflow.config.set_default_jitter(value)


def set_default_float(float_type: str) -> None:
    types = {
        "fp32": np.float32,
        "float32": np.float32,
        "fp64": np.float64,
        "float64": np.float64,
    }
    if float_type in types:
        gpflow.config.set_default_float(types[float_type])
    else:
        raise NotImplementedError(f"Unknown float type {float_type}")


def set_default_seed(seed: int) -> None:
    tf.random.set_seed(seed)
    np.random.seed(seed)


def get_default_float_str() -> str:
    types = {
        np.float32: "fp32",
        np.float64: "fp64",
    }

    return types[gpflow.config.default_float()]


def get_default_float() -> str:
    return gpflow.config.default_float()


@singledispatch
def create_model(model_cfg: ModelConfig, data: Data):
    raise NotImplementedError()


@singledispatch
def create_kernel(cfg: KernelConfig, data: Data):
    raise NotImplementedError()


@singledispatch
def optimize(
    model: gpflow.models.BayesianModel,
    dataset: Tuple[Data, Data],
    num_steps: int,
    logger: Logger,
    optimizer: str = None,
):
    raise NotImplementedError()


@singledispatch
def save(model: gpflow.models.BayesianModel, logdir: str):
    raise NotImplementedError()


@singledispatch
def load(model: gpflow.models.BayesianModel, filepath: str):
    raise NotImplementedError()


def model_parameters(model: gpflow.models.BayesianModel) -> Dict[str, np.ndarray]:
    return {k: v.numpy() for k, v in parameter_dict(model).items()}


@singledispatch
def metrics_fn(
    model: gpflow.models.BayesianModel, datasets: Tuple[Data, Data]
) -> Callable[[], Dict[str, float]]:
    raise NotImplementedError()


# Implementations


def _default_value():
    dfloat = gpflow.config.default_float()
    if dfloat in (np.float32, tf.float32):
        return 5e-3
    return 1e-6


def _default_positive():
    return positive(_default_value())


@create_kernel.register
def _create_kernel(cfg: SquaredExponentialConfig, data: Data):
    params = cfg.params(data)
    variance = params["variance"]
    lengthscales = params["lengthscales"]
    k = gpflow.kernels.SquaredExponential(variance=variance, lengthscales=lengthscales)
    k.variance = Parameter(k.variance, transform=_default_positive())
    k.lengthscales = Parameter(k.lengthscales, transform=_default_positive())
    return k


@create_kernel.register
def _create_kernel(cfg: Matern32Config, data: Data):
    params = cfg.params(data)
    variance = params["variance"]
    lengthscales = params["lengthscales"]
    k = gpflow.kernels.Matern32(variance=variance, lengthscales=lengthscales)
    k.variance = Parameter(k.variance, transform=_default_positive())
    k.lengthscales = Parameter(k.lengthscales, transform=_default_positive())
    return k


@create_model.register
def _create_model(model_cfg: GPRConfig, data: Data):
    kernel = create_kernel(model_cfg.kernel, data)
    params = model_cfg.params(data)
    variance = params["noise_variance"]
    mean = gpflow.mean_functions.Constant()
    return GPR(data, kernel, noise_variance=variance, mean_function=mean)


def handle_sgpr_params(model_cfg: SGPR, data: Data) -> Dict[str, Any]:
    kernel = create_kernel(model_cfg.kernel, data)
    params = model_cfg.params(data)
    inducing_variable = params["inducing_variable"](kernel)
    return params, kernel, inducing_variable


@create_model.register
def _create_model(model_cfg: SGPRConfig, data: Data):
    params, kernel, inducing_variable = handle_sgpr_params(model_cfg, data)
    mean = gpflow.mean_functions.Constant()
    model = SGPR(
        data,
        kernel,
        inducing_variable,
        noise_variance=params["noise_variance"],
        mean_function=mean,
    )
    return model


@create_model.register
def _create_model(model_cfg: SGPRN2MConfig, data: Data):
    params, kernel, inducing_variable = handle_sgpr_params(model_cfg, data)
    mean = gpflow.mean_functions.Constant()
    model = SGPRN2M(
        data,
        kernel,
        inducing_variable,
        noise_variance=params["noise_variance"],
        mean_function=mean,
    )
    return model


@create_model.register
def _create_model(model_cfg: CGLBConfig, data: Data):
    params, kernel, inducing_variable = handle_sgpr_params(model_cfg, data)
    m = CGLB(
        data,
        kernel,
        inducing_variable,
        max_error=params["max_error"],
        joint_optimization=params["joint_optimization"],
        vzero=params["vzero"],
        noise_variance=params["noise_variance"],
        mean_function=gpflow.mean_functions.Constant(),
    )
    m.likelihood = gpflow.likelihoods.Gaussian(variance_lower_bound=_default_value())
    return m


@create_model.register
def _create_model(model_cfg: CGLBN2MConfig, data: Data):
    params, kernel, inducing_variable = handle_sgpr_params(model_cfg, data)
    m = CGLBN2M(
        data,
        kernel,
        inducing_variable,
        max_error=params["max_error"],
        joint_optimization=params["joint_optimization"],
        vzero=params["vzero"],
        noise_variance=params["noise_variance"],
        mean_function=gpflow.mean_functions.Constant(),
    )
    m.likelihood = gpflow.likelihoods.Gaussian(variance_lower_bound=_default_value())
    return m


@create_model.register
def _create_model(model_cfg: CGLBNM2Config, data: Data):
    params, kernel, inducing_variable = handle_sgpr_params(model_cfg, data)
    m = CGLBNM2(
        data,
        kernel,
        inducing_variable,
        max_error=params["max_error"],
        joint_optimization=params["joint_optimization"],
        vzero=params["vzero"],
        noise_variance=params["noise_variance"],
        mean_function=gpflow.mean_functions.Constant(),
    )
    m.likelihood = gpflow.likelihoods.Gaussian(variance_lower_bound=_default_value())
    return m



@optimize.register
def _optimize(
    model: gpflow.models.BayesianModel,
    dataset: Tuple[Data, Data],
    num_steps: int,
    logger: Logger,
    optimizer: str = None,
):
    loss_fn: Callable = model.training_loss_closure(compile=jit_compile())
    variables = model.trainable_variables

    loss_fn()

    if optimizer is None or optimizer == "scipy":
        opt = gpflow.optimizers.Scipy()

        loss_fn()

        def _optimize_fn(params, maxiter: int, ftol=0.0, gtol=0.0):
            return opt.minimize(
                closure=loss_fn,
                variables=params,
                method="L-BFGS-B",
                options=dict(maxiter=maxiter, ftol=ftol, gtol=gtol),
                compile=jit_compile(),
                step_callback=logger,
            )

        logger.timer.reset()
        logger.timer.start()

        # First attempt 1.
        res1 = _optimize_fn(variables, num_steps)
        next_num_steps = num_steps - res1.nit
        print("Scipy Result 1: ")
        print(res1)

        # Second attempt 2.
        res2 = _optimize_fn(variables, next_num_steps)
        next_num_steps = next_num_steps - res2.nit
        print("Scipy Result 2: ")
        print(res2)

    elif optimizer.startswith("adam"):
        lr_str = optimizer.split("_", maxsplit=1)[1]
        lr = float(lr_str)
        opt = tf.optimizers.Adam(lr)

        def minimize_fn():
            opt.minimize(loss_fn, variables)

        if jit_compile():
            minimize_fn = tf.function(minimize_fn)

        logger.timer.reset()
        logger.timer.start()

        for i in range(num_steps):
            minimize_fn()
            logger(i)


@save.register
def _save(model: gpflow.models.BayesianModel, logdir: str):
    os.makedirs(logdir, exist_ok=True)
    model_params = model_parameters(model)
    with open(Path(logdir, "model.json"), "w") as file:
        json_tricks.dump(model_params, file)


@load.register
def _load(model: gpflow.models.BayesianModel, filepath: str):
    with open(filepath, "r") as file:
        loaded_params = json_tricks.load(file)
    model_param_keys = set(parameter_dict(model).keys())
    difference = model_param_keys.difference(loaded_params.keys())
    intersection = model_param_keys.intersection(loaded_params.keys())
    if len(difference) != 0:
        warnings.warn(f"Cannot load some parameters: {difference}")
    params = {k: loaded_params[k] for k in intersection}

    # TODO: depricate this condition
    if ".mean_function.c" in params:
        value = params[".mean_function.c"]
        params[".mean_function.c"] = np.atleast_1d(value)

    multiple_assign(model, params)
    return model


@metrics_fn.register
def _metrics_fn(model: GPR, datasets: Tuple[Data, Data]):
    rmse_lpd_metrics = _rmse_and_lpd_func(model, datasets)

    @tf.function
    def gpr_metrics() -> Dict[str, Tensor]:
        lml = model.log_marginal_likelihood()
        return dict(lml=lml, loss=-lml)

    return lambda: metric.call_metric_fns(gpr_metrics, rmse_lpd_metrics)


@metrics_fn.register
def _compute_metrics(model: SGPR, datasets: Tuple[Data, Data]):
    rmse_lpd_metrics = _rmse_and_lpd_func(model, datasets)

    @tf.function
    def sgpr_metrics() -> Dict[str, Tensor]:
        elbo = model.elbo()
        upper_bound = model.upper_bound()
        return dict(elbo=elbo, titsias_upper_bound=upper_bound, loss=-elbo)

    return lambda: metric.call_metric_fns(sgpr_metrics, rmse_lpd_metrics)


@metrics_fn.register
def _compute_metrics(model: CGLB, datasets: Tuple[Data, Data]):
    rmse_lpd_metrics = _rmse_and_lpd_func(model, datasets)

    def cglb_cg_params() -> Dict[str, np.ndarray]:
        steps = model.cg_steps.numpy()
        error = model.cg_residual_error.numpy()
        return {"cg/steps": steps, "cg/error": error}

    @tf.function
    def cglb_metrics() -> Dict[str, Tensor]:
        cglb = model.maximum_log_likelihood_objective()
        elbo = model.elbo()
        upper_bound = model.upper_bound()
        return dict(elbo=elbo, titsias_upper_bound=upper_bound, cg_lower_bound=cglb, loss=-cglb)

    return lambda: metric.call_metric_fns(cglb_cg_params, cglb_metrics, rmse_lpd_metrics)


def _err_and_logdensity(
    model: Union[GPR, SGPR], datasets: Tuple[Data, Data]
) -> Tuple[PairTensor, PairTensor]:
    train, test = datasets
    num = tf.shape(train[0])[0]
    x, y = (
        tf.concat([train[0], test[0]], 0),
        tf.concat([train[1], test[1]], 0),
    )
    predictions, variance = model.predict_f(x)
    err = y - predictions
    logden = model.likelihood.predict_log_density(predictions, variance, y)
    return (err[:num], err[num:]), (logden[:num], logden[num:])


def _rmse_and_lpd_func(model, datasets):
    @tf.function
    def optimized_func() -> Tuple[PairTensor, PairTensor]:
        return _err_and_logdensity(model, datasets)

    return metric.rmse_and_lpd_fn(optimized_func)
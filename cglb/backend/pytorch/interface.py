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

import os
from dataclasses import asdict
from functools import singledispatch
from pathlib import Path
from typing import Optional, Tuple

import gpytorch
import json_tricks
import numpy as np
import pykeops
import torch

# from gpytorch.beta_features import checkpoint_kernel
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import ScaleKernel
from gpytorch.models import ExactGP
from gpytorch.settings import (
    deterministic_probes,
    max_preconditioner_size,
    max_lanczos_quadrature_iterations,
)

from .. import metric
from ..callbacks import Logger
from ..config import (
    CGLBConfig,
    ExactGPConfig,
    KernelConfig,
    Matern32Config,
    ModelConfig,
    SGPRConfig,
    SquaredExponentialConfig,
)
from .conjugate_gradient import ConjugateGradient
from .lbfgs import FullBatchLBFGS
from .models import CGLB, GPR, SGPR, LowerBoundCG, PredictCG, log_density
from .optimizer import Scipy

__USE_KEOPS__ = True


__all__ = ["create_kernel", "create_model", "optimize", "save", "load", "metrics_fn"]


# Type stubs

Tensor = torch.Tensor
PairTensor = Tuple[Tensor, Tensor]
Data = Tuple[np.ndarray, np.ndarray]


def configure_backend(
    logdir: Optional[str] = None,
    keops: Optional[bool] = None,
    **kwargs,
):
    assert logdir is not None
    assert keops is not None

    global __USE_KEOPS__
    __USE_KEOPS__ = keops

    import tensorflow as tf

    devname = "CPU"
    devices = tf.config.get_visible_devices(devname)
    tf.config.set_visible_devices(devices, devname)
    tf.config.set_visible_devices([], "GPU")

    if __USE_KEOPS__:
        pykeops.set_bin_folder(f"{logdir}/keops/")
        pykeops.clean_pykeops()

    gpytorch.settings.max_preconditioner_size._set_value(_prec_size())

def set_default_jitter(jitter):
    gpytorch.settings.cholesky_jitter._set_value(jitter)


def set_default_float(float_type: str) -> None:
    types = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp64": torch.float64,
        "float64": torch.float64,
    }
    if float_type in types:
        torch.set_default_dtype(types[float_type])
    else:
        raise NotImplementedError(f"Unknown float type {float_type}")


def get_default_float_str() -> str:
    types = {
        torch.float32: "fp32",
        torch.float64: "fp64",
    }

    return types[torch.get_default_dtype()]


def get_default_float() -> np.dtype:
    return torch.tensor(1, dtype=torch.get_default_dtype()).detach().cpu().numpy().dtype


@singledispatch
def create_model(model_cfg: ModelConfig, data: Data):
    raise NotImplementedError()


@singledispatch
def create_kernel(cfg: KernelConfig, data: Data):
    raise NotImplementedError()


@singledispatch
def optimize(model: GPR, dataset: Tuple[Data, Data], num_steps: int, logdir: str, optimizer: str):
    raise NotImplementedError()


@singledispatch
def save(model: GPR, logdir: str):
    raise NotImplementedError()


@singledispatch
def load(model: GPR, filepath: str):
    raise NotImplementedError()


@singledispatch
def metrics_fn(model: GPR, dataset_bundle: Tuple[Data, Data]):
    raise NotImplementedError()


def model_parameters(model):
    noise = _numpy(model.likelihood.noise_covar.noise)[0]
    constant = _numpy(model.mean_module.constant)
    params = {
        ".likelihood.variance": noise,
        ".mean_function.c": constant,
    }

    if isinstance(model.covar_module, gpytorch.kernels.MultiDeviceKernel):
        kernel = model.covar_module.base_kernel
    else:
        kernel = model.covar_module

    if isinstance(kernel, gpytorch.kernels.InducingPointKernel):
        inducing_points = kernel.inducing_points
        params.update({".inducing_variable.Z": _numpy(inducing_points)})
        kernel = kernel.base_kernel

    lengthscale = _numpy(kernel.base_kernel.lengthscale)[0, :]
    outputscale = _numpy(kernel.outputscale)

    params.update(
        {
            ".kernel.lengthscales": lengthscale,
            ".kernel.variance": outputscale.squeeze(),
        }
    )

    return params


# Implementations


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    dtype = torch.get_default_dtype()
    tensor = torch.as_tensor(array, dtype=dtype)
    return tensor.to(_output_device())


def _output_device():
    n_devices = torch.cuda.device_count()
    device_name = "cuda" if n_devices > 0 else "cpu"
    return torch.device(device_name)


def _move_to_output_device(torch_object):
    device = _output_device()
    return torch_object.to(device)


def _dataset_to_tensor(data: Tuple):
    x = _to_tensor(data[0])
    y = _to_tensor(data[1])
    return _move_to_output_device(x), _move_to_output_device(y)


@create_kernel.register
def _create_kernel(cfg: SquaredExponentialConfig, data: Data):
    params = cfg.params(data)
    variance = _to_tensor(params["variance"])
    lengthscales = _to_tensor(params["lengthscales"])
    kernel_class = _load_kernel("rbf")
    rbf = kernel_class(ard_num_dims=len(lengthscales))
    rbf.lengthscale = _to_tensor(lengthscales)
    kernel = ScaleKernel(rbf)
    kernel.outputscale = _to_tensor(variance)
    return kernel


@create_kernel.register
def _create_kernel(cfg: Matern32Config, data: Data):
    params = cfg.params(data)
    variance = _to_tensor(params["variance"])
    lengthscales = _to_tensor(params["lengthscales"])
    kernel_class = _load_kernel("matern")
    matern = kernel_class(nu=1.5, ard_num_dims=len(lengthscales))
    matern.lengthscale = _to_tensor(lengthscales)
    kernel = ScaleKernel(matern, dtype=torch.float64)
    kernel.outputscale = _to_tensor(variance)
    return kernel


@create_model.register
def _create_model(model_cfg: ExactGPConfig, data: Data):
    n_devices = torch.cuda.device_count()
    output_device = _output_device()
    params = model_cfg.params(data)

    kernel = create_kernel(model_cfg.kernel, data)

    if n_devices != 0:
        kernel = gpytorch.kernels.MultiDeviceKernel(
            kernel, device_ids=range(n_devices), output_device=output_device
        )

    variance = _to_tensor(params["noise_variance"])
    # TODO: Default GPytorch version has high variance 1e-4
    positive = GreaterThan(1e-4)
    # positive = GreaterThan(1e-6)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=positive).to(
        output_device
    )
    likelihood.noise = variance
    data_tensors = (
        _to_tensor(data[0]),
        _to_tensor(data[1].reshape(-1)),
    )
    model = GPR(data_tensors, likelihood, kernel).to(output_device)
    model = model.cuda() if torch.cuda.is_available() else model
    return model


def _likelihood_and_kernel_for_sgpr(model_cfg: SGPRConfig, data: Data):
    params = model_cfg.params(data)
    variance = params["noise_variance"]

    output_device = _output_device()

    positive = GreaterThan(1e-6)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=positive).to(
        output_device
    )
    likelihood.noise = variance

    base_kernel = create_kernel(model_cfg.kernel, data)
    base_kernel = base_kernel.to(output_device)

    def init_kernel_fn(x1, x2, full_cov: bool = False):
        if x2 is None:
            x2 = x1
        x1 = _to_tensor(x1)
        x2 = _to_tensor(x2)
        diag = not full_cov
        return base_kernel(x1, x2, diag=diag).detach().cpu().numpy()

    params = model_cfg.params(data)
    inducing_variable = params["inducing_variable"](init_kernel_fn)
    inducing_variable = _to_tensor(inducing_variable).to(output_device)

    # NOTE: Does `MultiDeviceKernel` work with InducingPointKernel?
    n_devices = torch.cuda.device_count()
    if n_devices > 1:
        base_kernel = gpytorch.kernels.MultiDeviceKernel(
            base_kernel, device_ids=range(n_devices), output_device=output_device
        )

    kernel = gpytorch.kernels.InducingPointKernel(
        base_kernel, inducing_variable, likelihood=likelihood
    )

    return likelihood, kernel


@create_model.register
def _create_model(model_cfg: SGPRConfig, data: Data):
    likelihood, kernel = _likelihood_and_kernel_for_sgpr(model_cfg, data)
    data_tensors = (
        _to_tensor(data[0]),
        _to_tensor(data[1].reshape(-1)),
    )
    model = SGPR(data_tensors, likelihood, kernel).to(_output_device())
    return model


@create_model.register
def _create_model(model_cfg: CGLBConfig, data: Data):
    likelihood, kernel = _likelihood_and_kernel_for_sgpr(model_cfg, data)
    data_tensors = (
        _to_tensor(data[0]),
        _to_tensor(data[1].reshape(-1)),
    )
    model = CGLB(data_tensors, likelihood, kernel).to(_output_device())
    return model


@optimize.register
def _optimize(model: GPR, dataset: Tuple[Data, Data], num_steps: int, logger: Logger, optimizer: str):
    assert optimizer.startswith("adam")

    adam_lr = float(optimizer.split("_", maxsplit=1)[1])

    _clean_torch()
    _clean_keops()
    train_data = dataset[0]
    train_x = _to_tensor(train_data[0]).contiguous()
    train_y = _to_tensor(train_data[1].reshape(-1)).contiguous()

    train_x = _move_to_output_device(train_x)
    train_y = _move_to_output_device(train_y)

    nth = 10000
    train_x_nth = train_x[:nth]
    train_y_nth = train_y[:nth]

    model.set_train_data(train_x_nth, train_y_nth, strict=False)
    likelihood = _move_to_output_device(model.likelihood)

    model.train()
    likelihood.train()

    lbfgs = FullBatchLBFGS(model.parameters(), lr=0.1)
    adam = torch.optim.Adam(model.parameters(), lr=adam_lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    prec_size = _prec_size()
    # TODO:
    # part_size = 10000
    # with max_lanczos_quadrature_iterations(2)
    # with checkpoint_kernel(part_size), max_preconditioner_size(prec_size), deterministic_probes(
    #     state=True
    # ):

    # TODO: Change it back
    # with max_preconditioner_size(prec_size), deterministic_probes(state=True), max_lanczos_quadrature_iterations(2):
    with max_preconditioner_size(prec_size), deterministic_probes(state=True):

        def lbfgs_closure():
            lbfgs.zero_grad()
            distr = model(train_x_nth)
            loss = -mll(distr, train_y_nth)
            return loss

        counter = 0

        # First ten iterations with LBFGS and subset of the data
        lbfgs_loss = lbfgs_closure()
        lbfgs_loss.backward()

        logger.timer.reset()
        logger.timer.start()

        for i in range(10):
            options = {
                "closure": lbfgs_closure,
                "current_loss": lbfgs_loss,
                "max_ls": 10,
            }
            res = lbfgs.step(options)
            loss, fail = res[0], res[-1]
            # TODO: turn of logger temporarily
            logger(counter)
            counter += 1

    # Second ten iterations with Adam and subset of the data
    # with checkpoint_kernel(part_size), max_preconditioner_size(prec_size), deterministic_probes(
    #     state=False
    # ):

    # TODO: Change it back
    # with max_preconditioner_size(prec_size), deterministic_probes(state=False), max_lanczos_quadrature_iterations(2):
    with max_preconditioner_size(prec_size), deterministic_probes(state=False):

        for i in range(10):
            adam.zero_grad()
            distr = model(train_x_nth)
            loss = -mll(distr, train_y_nth)
            loss.backward()
            adam.step()
            # TODO: turn of logger temporarily
            logger(counter)
            counter += 1

    # N iterations with Adam on FULL dataset

    # TODO: Change learning rate to improve stability or delay divergence,
    #       however, it makes performace worse for gpytorch implementation as well
    # for g in adam.param_groups:
    #     g["lr"] = 0.01

    # TODO: `checkpoint_kernel` doesn't work as expected.
    # with checkpoint_kernel(part_size), max_preconditioner_size(prec_size), deterministic_probes(
    #     state=False
    # ):

    # TODO: Change it back
    # with max_preconditioner_size(prec_size), deterministic_probes(state=False):
    # with max_preconditioner_size(prec_size), deterministic_probes(state=False), max_lanczos_quadrature_iterations(2):
    with max_preconditioner_size(prec_size), deterministic_probes(state=False):

        model.set_train_data(train_x, train_y, strict=False)

        for i in range(num_steps):
            adam.zero_grad()
            distr = model(train_x)
            loss = -mll(distr, train_y)
            loss.backward()
            adam.step()
            logger(counter)
            counter += 1

        logger.holdout_interval = 1
        logger(counter) # record last step


@optimize.register
def _optimize(model: CGLB, dataset: Tuple[Data, Data], num_steps: int, logger: Logger, optimize: str):
    assert optimize == "scipy"

    _clean_torch()
    _clean_keops()
    # TODO:
    max_n = 10000
    gpytorch.settings.max_cholesky_size._set_value(max_n)
    # gpytorch.settings.max_eager_kernel_size._set_value(2)

    train_data = dataset[0]
    train_x = _to_tensor(train_data[0]).contiguous()
    train_y = _to_tensor(train_data[1].reshape(-1)).contiguous()

    train_x = _move_to_output_device(train_x)
    train_y = _move_to_output_device(train_y)

    train_data = (train_x, train_y)

    likelihood = _move_to_output_device(model.likelihood)
    # model.set_train_data(train_x, train_y, strict=False)

    model.train()
    likelihood.train()

    lbfgs = Scipy()
    lower_bound = LowerBoundCG(model)

    def lbfgs_closure() -> Tensor:
        loss = -lower_bound(train_data)
        logger.log_for_feval(**asdict(model.cg_stats))
        return loss

    def step_callback(*args):
        lower_bound.cached_v_vec = False
        logger(*args)

    def optimize_fn(params, maxiter: int, ftol: float = 0.0, gtol: float = 0.0, disp: bool = False):
        options = dict(maxiter=maxiter, ftol=ftol, gtol=gtol, disp=disp)
        return lbfgs.minimize(
            lbfgs_closure,
            params,
            options=options,
            step_callback=step_callback,
        )

    params = list(model.parameters())

    # Pre-compile time
    with logger.no_recording():
        _loss = lbfgs_closure()
        _grads = torch.autograd.grad(_loss, params)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # logger.holdout_interval = -1  # Uncomment if you don't want to log values
    logger.timer.reset()
    logger.timer.start()

    # 1) First Scipy optimisation. Annoyingly, Scipy L-BFGS stops early
    # (might be a bug which is fixed in the following versions)
    result1 = optimize_fn(params, num_steps)
    next_num_steps = num_steps - result1.nit
    print("Scipy Result 1: ")
    print(result1)

    # 2) Repeat the Scipy step
    if next_num_steps <= 0:
        return

    result2 = optimize_fn(params, next_num_steps)
    next_num_steps = next_num_steps - result2.nit
    print("Scipy Result 2: ")
    print(result2)

    # 3) Repeat the Scipy step w/o inducing points
    if next_num_steps <= 0:
        return

    ips = model.covar_module.inducing_points
    ips.required_grad = False
    params = [p for p in model.parameters() if id(p) != id(ips)]

    result3 = optimize_fn(params, next_num_steps)
    next_num_steps = next_num_steps - result3.nit
    print("Scipy Result 3: ")
    print(result3)

    # 4) Repeat the Scipy again w/o
    if next_num_steps <= 0:
        return

    result4 = optimize_fn(params, next_num_steps)
    next_num_steps = next_num_steps - result4.nit
    print("Scipy Result 4: ")
    print(result4)


@save.register
def _save(model: GPR, logdir: str):
    os.makedirs(logdir, exist_ok=True)
    params = model_parameters(model)
    with open(Path(logdir, "model.json"), "w") as file:
        json_tricks.dump(params, file)


@load.register
def _load(model: ExactGP, filepath: str):
    param_dict = torch.load(filepath)
    model.load_state_dict(param_dict)
    return model


@metrics_fn.register
def _compute_metrics(model: GPR, dataset_bundle: Tuple[Data, Data]):
    def gpr_metrics():
        training_status = model.training
        model.train()
        with torch.no_grad():
            n = model.train_inputs[0].shape[0]
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
            loss = -mll(model(model.train_inputs[0]), model.train_targets) * n
        model.train(training_status)
        return dict(loss=_numpy(loss))

    train, _ = dataset_bundle
    x_full, y_full = _merge_datasets(dataset_bundle)
    y_full = y_full.reshape(-1)
    total_data_size = int(x_full.shape[0])

    def error_and_logdensity():
        lpds, errs = list(), list()
        max_batch = int(1e5)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(0, total_data_size, max_batch):
                x_batch = x_full[i : i + max_batch]
                y_batch = y_full[i : i + max_batch]

                y_distr = model(x_batch)
                y_pred = y_distr.mean
                lpd = model.likelihood.log_marginal(y_batch, y_distr)
                lpd = _numpy(lpd)
                err = _numpy(y_full - y_pred)

                lpds.append(lpd)
                errs.append(err)

        err = np.concatenate(errs, axis=0)
        lpd = np.concatenate(lpds, axis=0)

        n = train[0].shape[0]
        return (err[:n], err[n:]), (lpd[:n], lpd[n:])

    rmse_lpd_metrics = metric.rmse_and_lpd_fn(error_and_logdensity)

    return lambda: _call_metrics_callbacks(model, gpr_metrics, rmse_lpd_metrics)


@metrics_fn.register
def _compute_metrics(model: CGLB, dataset_bundle: Tuple[Data, Data]):
    def cglb_cg_params():
        if model.cg_stats is not None:
            steps = model.cg_stats.steps
            error = model.cg_stats.residual_error
            return {"cg/steps": _numpy(steps), "cg/error": _numpy(error)}
        return {}

    train, test = dataset_bundle
    data = _dataset_to_tensor(train)

    def cglb_metrics():
        with torch.no_grad():
            lower_bound = LowerBoundCG(
                model,
                use_cache=True,
                cached_v_vec_initial=True,
            )
            loss = -lower_bound(data)
            return dict(loss=_numpy(loss))

    x_full, y_full = _merge_datasets(dataset_bundle)
    y_full = y_full.reshape(-1, 1)
    total_data_size = int(x_full.shape[0])

    def error_and_logdensity():
        predict_f = PredictCG(model)

        lpds, errs = list(), list()
        max_batch = int(1e6)

        with torch.no_grad():
            for i in range(0, total_data_size, max_batch):
                f_mean, f_var = predict_f(x_full[i : i + max_batch])
                y_batch = y_full[i : i + max_batch]
                lpd = log_density(model, y_batch, f_mean, f_var)
                lpd = _numpy(lpd)
                err = _numpy(y_batch - f_mean)

                lpds.append(lpd)
                errs.append(err)

        err = np.concatenate(errs, axis=0)
        lpd = np.concatenate(lpds, axis=0)

        n = train[0].shape[0]
        return (err[:n], err[n:]), (lpd[:n], lpd[n:])

    rmse_lpd_metrics = metric.rmse_and_lpd_fn(error_and_logdensity)

    return lambda: _call_metrics_callbacks(model, cglb_cg_params, cglb_metrics, rmse_lpd_metrics)


def _merge_datasets(datasets: Tuple[Data, Data]) -> Data:
    x_train, y_train = _dataset_to_tensor(datasets[0])
    x_test, y_test = _dataset_to_tensor(datasets[1])
    x_full = torch.cat([x_train, x_test], axis=0)
    y_full = torch.cat([y_train, y_test], axis=0)
    return x_full, y_full


def _numpy(tensor: Tensor) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def _call_metrics_callbacks(model, *cbs):
    training_status = model.training
    model.eval()
    model.likelihood.eval()
    metrics = metric.call_metric_fns(*cbs)
    model.likelihood.train(training_status)
    model.train(training_status)
    return metrics


def _clean_torch():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _clean_keops():
    if __USE_KEOPS__:
        pykeops.clean_pykeops()


def _load_kernel(name: str):
    if __USE_KEOPS__:
        from gpytorch.kernels.keops import MaternKernel, RBFKernel

        if name == "matern":
            return MaternKernel
        elif name == "rbf":
            return RBFKernel
    else:
        from gpytorch.kernels import MaternKernel, RBFKernel

        if name == "matern":
            return MaternKernel
        elif name == "rbf":
            return RBFKernel
    raise ValueError("Unknown kernel type requested")


def _prec_size() -> int:
    return 100
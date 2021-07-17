import json
import numpy as np
from pathlib import Path
from typing import Text, Dict, Any, Type, Callable, Optional
from dataclasses import dataclass, field
import click

import json_tricks

from cglb_experiments.click_types import (
    Context,
    BackendType,
    GPRConfigType,
    SGPRConfigType,
    KernelConfigType,
    InducingVariableConfigType,
    DatasetType,
)
from cglb_experiments.datasets import DatasetBundle

from cglb.backend.callbacks import Logger
from cglb.backend.backend import Backend
from cglb.backend.config import (
    ModelConfig,
    GPRConfig,
    KernelConfig,
    SGPRConfig,
    CGLBConfig,
    CGLBN2MConfig,
    InducingVariableConfig,
)
from cglb_experiments.baselines import meanpred_baseline, linear_baseline


_default_logdir = "./logdir"


@dataclass(frozen=True)
class ExecuteContext:
    main_ctx: Context
    dataset: DatasetBundle
    callback_fn: Callable[[Any], Any]
    extra_args: Dict = field(default_factory=lambda: {})


@click.group()
@click.option("-b", "--backend", type=BackendType(), required=True)
@click.option("-t", "--float-type", type=click.Choice(["fp32", "fp64"]), default="fp32")
@click.option("-l", "--logdir", type=click.Path(file_okay=False), default=_default_logdir)
@click.option("-s", "--seed", type=int, default=0)
@click.option("--keops/--no-keops", default=True)
@click.pass_context
def main(
    ctx: click.Context, backend: Backend, float_type: str, logdir: str, seed: int, keops: bool
):
    logdir_path = Path(logdir).expanduser().resolve()
    logdir_path.mkdir(exist_ok=True, parents=True)
    logdir = str(logdir_path)
    backend.configure_backend(logdir=logdir, keops=keops)
    backend.set_default_float(float_type)
    backend.set_default_jitter(float_type)
    ctx.obj = Context(backend, seed, logdir)


def create_optimize_fn(
    backend: Backend,
    dataset_bundle: DatasetBundle,
    logdir: str,
    num_steps: int,
    seed: int,
    optimizer: str,
) -> Callable:
    def optimize_fn(model: Any):
        datasets = dataset_bundle.to_tuple()
        holdout_interval = 20

        metrics_fn = backend.metrics_fn(model, datasets)
        model_parameters_fn = lambda: backend.model_parameters(model)

        logger = Logger(
            logdir, metrics_fn, model_parameters_fn, holdout_interval, include_feval_log=True
        )
        backend.optimize(model, datasets, num_steps, logger, optimizer)
        backend.save(model, logdir)

        logs = logger.logs
        results = metrics_fn()
        results["id"] = logdir
        logs["id"] = logdir

        with open(Path(logdir, "results.json"), "w") as file:
            json_tricks.dump(results, file)

        with open(Path(logdir, "logs.json"), "w") as file:
            json_tricks.dump(logs, file)

    return optimize_fn


def create_metric_fn(
    backend: Backend, dataset_bundle: DatasetBundle, destination: Path
) -> Callable:
    def metric_fn(model: Any):
        metrics_fn = backend.metrics_fn(model, dataset_bundle.to_tuple())
        results = metrics_fn()
        results["id"] = str(destination.parent)
        np.save(destination, results)

    return metric_fn


def create_baseline_fn(dataset_bundle: DatasetBundle, logdir: str, baseline: str) -> Callable:
    def baseline_fn(model: Any):
        baseline_fns = {"linear": linear_baseline, "mean": meanpred_baseline}
        baseline_fn = baseline_fns[baseline]
        results = baseline_fn(dataset_bundle)
        results["id"] = baseline
        with open(Path(logdir, "results.json"), "w") as file:
            json_tricks.dump(results, file)

    return baseline_fn


_optimizer_choices = click.Choice(["scipy", "adam_0.1", "adam_0.01", "adam_0.001"])


@main.group()
@click.option("-n", "--num-steps", default=100, type=int)
@click.option("-d", "--dataset", type=DatasetType(), required=True)
@click.option("-o", "--optimizer", type=_optimizer_choices, default="scipy")
@click.pass_context
def train(ctx: click.Context, dataset: DatasetBundle, num_steps: int, optimizer: str):
    main_ctx: Context = ctx.obj
    backend = main_ctx.backend
    optimize_fn = create_optimize_fn(
        backend, dataset, main_ctx.logdir, num_steps, main_ctx.seed, optimizer
    )
    ctx.obj = ExecuteContext(main_ctx=main_ctx, dataset=dataset, callback_fn=optimize_fn)


@main.group()
@click.option("-d", "--dataset", type=DatasetType(), required=True)
@click.pass_context
def metric(ctx: click.Context, dataset: DatasetBundle):
    main_ctx: Context = ctx.obj
    backend = main_ctx.backend
    dst = Path(main_ctx.logdir, "metric.npy")
    predict_fn = create_metric_fn(backend, dataset, dst)
    ctx.obj = ExecuteContext(main_ctx=main_ctx, dataset=dataset, callback_fn=predict_fn)


@main.command()
@click.option("-d", "--dataset", type=DatasetType(), required=True)
@click.option("-k", "--kernel", type=KernelConfigType(), required=True)
@click.option("-p", "--param_file", type=click.Path(readable=True), required=True)
@click.pass_context
def gpr_metric(
    ctx: click.Context, dataset: DatasetBundle, kernel: Type[KernelConfig], param_file: str
):
    main_ctx: Context = ctx.obj
    backend = main_ctx.backend
    params_path = Path(param_file)
    dst = Path(params_path.parent, "gpr_metric.npy")
    gpr_metric_fn = create_metric_fn(backend, dataset, dst)
    model = GPRConfig(kernel())
    ctx.obj = ExecuteContext(main_ctx=main_ctx, dataset=dataset, callback_fn=gpr_metric_fn)
    _execute_cb_on_model(ctx, model, param_file)


# @main.group()
@main.command()
@click.option("-d", "--dataset", type=DatasetType(), required=True)
@click.argument("baseline", type=click.Choice(["mean", "linear"]))
@click.pass_context
def baseline(ctx: click.Context, baseline: str, dataset: DatasetBundle):
    main_ctx: Context = ctx.obj
    predict_fn = create_baseline_fn(dataset, main_ctx.logdir, baseline)
    exec_ctx = ExecuteContext(main_ctx=main_ctx, dataset=dataset, callback_fn=predict_fn)
    exec_ctx.callback_fn(None)


gpr_options = [
    click.option("-m", "--model-class", type=GPRConfigType(), required=True),
    click.option("-k", "--kernel", type=KernelConfigType(), required=True),
    click.option("-p", "--param_file", type=click.Path(readable=True), required=False),
]


sgpr_options = [
    click.option("-m", "--model-class", type=SGPRConfigType(), required=True),
    click.option("-k", "--kernel", type=KernelConfigType(), required=True),
    click.option("-i", "--inducing-variable", type=InducingVariableConfigType(), required=True),
    click.option("-M", "--num-inducing-variables", default=100, type=int),
    click.option("-p", "--param_file", type=click.Path(readable=True)),
]

cglb_options = [
    click.option("-m", "--model-class", type=SGPRConfigType(), required=True),
    click.option("-k", "--kernel", type=KernelConfigType(), required=True),
    click.option("-i", "--inducing-variable", type=InducingVariableConfigType(), required=True),
    click.option("-M", "--num-inducing-variables", default=100, type=int),
    click.option("-p", "--param_file", type=click.Path(readable=True)),
    click.option("-e", "--max_error", type=float, default=1.0),
    click.option("--vjoint/--no-vjoint", default=False),
    click.option("--vzero/--no-vzero", default=False),
]


def add_options(options):
    def _wrapper(f):
        for option in reversed(options):
            f = option(f)
        return f

    return _wrapper


def _execute_cb_on_model(
    ctx: click.Context,
    model_config: ModelConfig,
    params_file: Optional[str] = None,
):
    exec_ctx: ExecuteContext = ctx.obj
    main_ctx = exec_ctx.main_ctx
    model = main_ctx.backend.create_model(model_config, exec_ctx.dataset.train)
    if params_file:
        model = main_ctx.backend.load(model, params_file)
    exec_ctx.callback_fn(model)


def _execute_cb_sgpr(
    ctx: click.Context,
    model_class: Type[SGPRConfig],
    kernel: Type[KernelConfig],
    inducing_variable: Type[InducingVariableConfig],
    num_inducing_variables: int,
    param_file: str,
):
    k: KernelConfig = kernel()
    iv: InducingVariableConfig = inducing_variable(num_inducing_variables)
    m: SGPRConfig = model_class(k, iv)
    _execute_cb_on_model(ctx, m, param_file)


def _execute_cb_cglb(
    ctx: click.Context,
    model_class: Type[CGLBConfig],
    kernel: Type[KernelConfig],
    inducing_variable: Type[InducingVariableConfig],
    num_inducing_variables: int,
    param_file: str,
    max_error: float,
    vjoint: bool,
    vzero: bool,
):
    k: KernelConfig = kernel()
    iv: InducingVariableConfig = inducing_variable(num_inducing_variables)
    m: SGPRConfig = model_class(k, iv, max_error, vjoint, vzero)
    _execute_cb_on_model(ctx, m, param_file)


def _execute_cb_cglbn2m(
    ctx: click.Context,
    model_class: Type[CGLBN2MConfig],
    kernel: Type[KernelConfig],
    inducing_variable: Type[InducingVariableConfig],
    num_inducing_variables: int,
    param_file: str,
    max_error: float,
    vjoint: bool,
    vzero: bool,
):
    k: KernelConfig = kernel()
    iv: InducingVariableConfig = inducing_variable(num_inducing_variables)
    m: SGPRConfig = model_class(k, iv, max_error, vjoint, vzero)
    _execute_cb_on_model(ctx, m, param_file)


def _execute_cb_gpr(
    ctx: click.Context,
    model_class: Type[GPRConfig],
    kernel: Type[KernelConfig],
    param_file: str,
):
    k: KernelConfig = kernel()
    m: GPRConfig = model_class(k)
    _execute_cb_on_model(ctx, m, param_file)


action_configs = [
    ("sgpr", (sgpr_options, _execute_cb_sgpr)),
    ("sgprn2m", (sgpr_options, _execute_cb_sgpr)),
    ("cglb", (cglb_options, _execute_cb_cglb)),
    ("cglbn2m", (cglb_options, _execute_cb_cglbn2m)),
    ("cglbnm2", (cglb_options, _execute_cb_cglbn2m)),
    ("gpr", (gpr_options, _execute_cb_gpr)),
]

# Train functions
train_actions = {
    k: train.command(k)(add_options(o)(click.pass_context(c))) for k, (o, c) in action_configs
}


# Metric functions
metric_actions = {
    k: metric.command(k)(add_options(o)(click.pass_context(c))) for k, (o, c) in action_configs
}


if __name__ == "__main__":
    main()

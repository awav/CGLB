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

from typing import Dict, Any, Tuple, Type
from dataclasses import dataclass
from functools import singledispatch
import numpy as np
import click

from cglb_experiments.datasets import get_dataset, DatasetBundle
from cglb.backend.backend import Backend, BACKENDS
from cglb.backend.config import (
    Config,
    ModelConfig,
    GPRConfig,
    GPR_CONFIGS,
    SGPRConfig,
    SGPR_CONFIGS,
    KernelConfig,
    KERNEL_CONFIGS,
    InducingVariableConfig,
    INDUCING_VARIABLE_CONFIGS,
)


__all__ = [
    "experiment_id",
    "GPRConfigType",
    "SGPRConfigType",
    "KernelConfigType",
    "InducingVariableConfigType",
    "DatasetType",
]


@dataclass(frozen=True)
class Context:
    backend: Backend
    seed: int
    logdir: str


def find_type_in_dict(param_type, key: str, store: Dict[str, Any], *fail_args) -> Type[Any]:
    if key in store:
        return store[key]
    param_type.fail(f"'{key}' is not a valid value", *fail_args)


class ParamType(click.ParamType):
    name: str = None
    class_type: Type[Any] = None
    class_dict: Dict[str, Type] = {}

    def convert(self, value: str, param, ctx):
        if value in self.class_dict:
            return self.class_dict[value]
        self.fail(f"'{value}' is not a valid value", param, ctx)


class BackendType(ParamType):
    name = "backend"
    class_type: Type[Backend] = Backend
    class_dict: Dict[str, Type[Backend]] = BACKENDS


class GPRConfigType(ParamType):
    name = "gpr_config"
    class_type: Type[GPRConfig] = GPRConfig
    class_dict: Dict[str, Type[GPRConfig]] = GPR_CONFIGS

    # def convert(self, value: str, param, ctx) -> Type[ModelConfig]:
    #     return find_type_in_dict(self, value, GPR_CONFIGS, param, ctx)


class SGPRConfigType(ParamType):
    name = "sgpr_config"
    class_type: Type[SGPRConfig] = SGPRConfig
    class_dict: Dict[str, Type[SGPRConfig]] = SGPR_CONFIGS

    # def convert(self, value: str, param, ctx) -> Type[ModelConfig]:
    #     return find_type_in_dict(self, value, SGPR_CONFIGS, param, ctx)


class KernelConfigType(ParamType):
    name = "kernel_config"
    class_type: Type[KernelConfig] = KernelConfig
    class_dict: Dict[str, Type[KernelConfig]] = KERNEL_CONFIGS

    # def convert(self, value: str, param, ctx) -> Type[KernelConfig]:
    #     return find_type_in_dict(self, value, KERNEL_CONFIGS, param, ctx)


class InducingVariableConfigType(ParamType):
    name = "inducing_variable_config"
    class_type: Type[InducingVariableConfig] = InducingVariableConfig
    class_dict: Dict[str, Type[InducingVariableConfig]] = INDUCING_VARIABLE_CONFIGS

    # def convert(self, value: str, param, ctx) -> Type[KernelConfig]:
    #     return find_type_in_dict(self, value, INDUCING_VARIABLE_CONFIGS, param, ctx)


class DatasetType(click.ParamType):
    name = "dataset"

    def convert(self, value: str, param, ctx) -> DatasetBundle:
        try:
            dtype = ctx.obj.backend.get_default_float()
            split = ctx.obj.seed
            return get_dataset(value, dtype, split=split)
        except Exception:
            self.fail(f"'{value}' is not a valid value", param, ctx)


@singledispatch
def experiment_id(model_cfg: ModelConfig, *args, **kwargs):
    raise NotImplementedError()


@experiment_id.register
def _experiment_id(model_cfg: GPRConfig, kernel_cfg: KernelConfig, **kwargs):
    model_name: str = _name_strip_config(model_cfg)
    kernel_name: str = _name_strip_config(kernel_cfg)
    head: str = f"{model_name}-{kernel_name}"
    tail: str = "-".join([f"{k}_{v}" for (k, v) in kwargs])
    return head if tail == "" else f"{head}-{tail}"


@experiment_id.register
def _experiment_id(
    model_cfg: SGPRConfig,
    kernel_cfg: KernelConfig,
    inducing_variable_cfg: InducingVariableConfig,
    **kwargs,
):
    model_name: str = _name_strip_config(model_cfg)
    kernel_name: str = _name_strip_config(kernel_cfg)
    inducing_variable_name: str = _name_strip_config(inducing_variable_cfg)
    head: str = f"{model_name}-{kernel_name}-{inducing_variable_name}"
    tail: str = "-".join([f"{k}_{v}" for (k, v) in kwargs.items()])
    return f"{head}-{tail}"


def _name_strip_config(cfg: Config) -> str:
    return cfg.__class__.__name__.rstrip("Config")

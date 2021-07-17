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

from typing import Dict, Union, Tuple, Callable
from abc import ABC, abstractmethod
from functools import partial
import dataclasses
import numpy as np
import robustgp

__all__ = [
    "Config",
    "ModelConfig",
    "KernelConfig",
    "SquaredExponentialConfig",
    "Matern32Config",
    "CGLBConfig",
    "CGLBN2MConfig",
    "CGLBNM2Config",
    "SGPRN2MConfig",
    "GPRConfig",
    "SGPRConfig",
    "GPR_CONFIGS",
    "SGPR_CONFIGS",
    "KERNEL_CONFIGS",
    "INDUCING_VARIABLE_CONFIGS",
]

Data = Tuple[np.ndarray, np.ndarray]

dataclass_frozen = partial(dataclasses.dataclass, frozen=True)


class Config:
    def params(self, **kwargs) -> Dict[str, Union[float, np.ndarray]]:
        pass


@dataclass_frozen
class ModelConfig(Config):
    pass


@dataclass_frozen
class InducingVariableConfig(Config):
    num_variables: int

    def params(self, data: Data) -> Dict[str, Union[float, np.ndarray]]:
        ...

    def init(self, data: Data, kernel_fn: Callable[[np.ndarray], np.ndarray]):
        init_fn = robustgp.ConditionalVariance(sample=False)
        iv, _ = init_fn(data[0], self.num_variables, kernel_fn)
        return iv


class KernelConfig(Config):
    pass


@dataclass_frozen
class SquaredExponentialConfig(KernelConfig):
    def params(self, data: Data) -> Dict[str, Union[float, np.ndarray]]:
        vecdim = data[0].shape[-1]
        return {"variance": 1.0, "lengthscales": np.repeat(1.0, vecdim)}


@dataclass_frozen
class Matern32Config(SquaredExponentialConfig):
    pass


@dataclass_frozen
class GPRConfig(ModelConfig):
    kernel: KernelConfig

    def params(self, data: Data) -> Dict[str, Union[float, np.ndarray]]:
        return {"noise_variance": 1.0}


@dataclass_frozen
class ExactGPConfig(GPRConfig):
    ...


@dataclass_frozen
class SGPRConfig(ModelConfig):
    kernel: KernelConfig
    inducing_variable: InducingVariableConfig

    def params(self, data: Data) -> Dict[str, Union[float, np.ndarray, Callable]]:
        inducing_variable_fn = partial(self.inducing_variable.init, data)
        return {
            "noise_variance": 1.0,
            "inducing_variable": inducing_variable_fn,
        }


@dataclass_frozen
class CGLBConfig(SGPRConfig):
    max_error: float = 1.0
    joint_optimization: bool = False
    vzero: bool = False

    def params(self, data: Data) -> Dict[str, Union[float, np.ndarray]]:
        param_dict = super().params(data)
        param_dict["max_error"] = self.max_error
        param_dict["joint_optimization"] = self.joint_optimization
        param_dict["vzero"] = self.vzero
        return param_dict


@dataclass_frozen
class CGLBN2MConfig(CGLBConfig):
    pass


@dataclass_frozen
class CGLBNM2Config(CGLBConfig):
    pass


@dataclass_frozen
class SGPRN2MConfig(SGPRConfig):
    pass


GPR_CONFIGS = {
    "gpr": GPRConfig,
    "exactgp": ExactGPConfig,
}

SGPR_CONFIGS = {
    "sgpr": SGPRConfig,
    "cglb": CGLBConfig,
    "sgprn2m": SGPRN2MConfig,
    "cglbn2m": CGLBN2MConfig,
    "cglbnm2": CGLBNM2Config,
}

KERNEL_CONFIGS = {
    "SquaredExponential": SquaredExponentialConfig,
    "Matern32": Matern32Config,
    # Aliases
    "mat32": Matern32Config,
    "rbf": SquaredExponentialConfig,
}

INDUCING_VARIABLE_CONFIGS = {
    "InducingVariable": InducingVariableConfig,
    "ConditionalVariance": InducingVariableConfig,
    # Aliases
    "iv": InducingVariableConfig,
    "cv": InducingVariableConfig,
}

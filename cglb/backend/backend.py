from abc import ABC, abstractmethod
from typing import Tuple, TypeVar, Dict
from .callbacks import Logger
from .config import KernelConfig, ModelConfig
from . import tensorflow, pytorch

import numpy as np


Data = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[Data, Data]


__all__ = ["Backend", "Pytorch", "Tensorflow", "Jax", "BACKENDS"]


Interfaces = TypeVar("Interfaces", type(pytorch.interface), type(tensorflow.interface))


class Backend(ABC):
    @staticmethod
    @abstractmethod
    def interface() -> Interfaces:
        pass

    @classmethod
    def configure_backend(cls, **kwargs):
        return cls.interface().configure_backend(**kwargs)

    @classmethod
    def create_kernel(cls, cfg: KernelConfig, data: Data):
        return cls.interface().create_kernel(cfg, data)

    @classmethod
    def create_model(cls, model_cfg: ModelConfig, data: Data):
        return cls.interface().create_model(model_cfg, data)

    @classmethod
    def model_parameters(cls, model) -> Dict[str, np.ndarray]:
        return cls.interface().model_parameters(model)

    @classmethod
    def optimize(cls, model, dataset: Dataset, num_steps: int, logger: Logger, optimizer: str):
        return cls.interface().optimize(model, dataset, num_steps, logger, optimizer)

    @classmethod
    def save(cls, model, logdir: str):
        return cls.interface().save(model, logdir)

    @classmethod
    def load(cls, model, filepath: str):
        return cls.interface().load(model, filepath)

    @classmethod
    def metrics_fn(cls, model, dataset_bundle: Tuple[Data, Data]):
        return cls.interface().metrics_fn(model, dataset_bundle)

    @classmethod
    def set_default_float(cls, float_type: str):
        return cls.interface().set_default_float(float_type)

    @classmethod
    def set_default_jitter(cls, float_type: str):
        value = 1e-5 if float_type == "fp32" else 1e-6
        return cls.interface().set_default_jitter(value)

    @classmethod
    def set_seed(cls, seed: int):
        return cls.interface().set_default_seed(seed)

    @classmethod
    def get_default_float_str(cls):
        return cls.interface().get_default_float_str()

    @classmethod
    def get_default_float(cls):
        return cls.interface().get_default_float()


class Tensorflow(Backend):
    @staticmethod
    def interface():
        return tensorflow.interface


class Pytorch(Backend):
    @staticmethod
    def interface():
        return pytorch.interface


# class Jax(Backend):
#     _backend = jax


BACKENDS = {
    "tensorflow": Tensorflow,
    "torch": Pytorch,
    "tf": Tensorflow,
    # "jax": Jax,
}

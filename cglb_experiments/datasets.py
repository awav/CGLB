from typing import Text, Tuple, TypeVar, Type
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from robustgp_experiments import utils


Dataset = TypeVar("Dataset", bound=Tuple[np.ndarray, np.ndarray])


@dataclass(frozen=True)
class DatasetBundle:
    name: Text
    train: Dataset
    test: Dataset

    def to_tuple(self):
        return (self.train, self.test)


def norm(x: np.ndarray) -> np.ndarray:
    """Normalise array with mean and variance."""
    mu = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True) + 1e-6
    return (x - mu) / std, mu, std


def norm_dataset(data: Dataset) -> Dataset:
    """Normalise dataset tuple."""
    return norm(data[0]), norm(data[1])


def get_dataset(
    name: Text,
    dtype: Type[np.float],
    normalize: bool = True,
    prop: float = 0.67,
    split: int = 0,
) -> DatasetBundle:
    """Return a dataset using a name."""
    dataset_fn = getattr(utils.data, name)
    if name == "snelson1d":
        cache_path = str(Path("~/.datasets").expanduser().resolve())
        data = dataset_fn(cache_path)
    else:
        data = dataset_fn(split=split, prop=prop)

    def _to_dtype(x, y):
        return (np.array(x, dtype=dtype), np.array(y, dtype=dtype))

    if isinstance(data, tuple):
        # This normalization is only correct when test=train
        train, test = data[0], data[1]
    elif isinstance(data, utils.data.Dataset):
        train, test = (data.X_train, data.Y_train), (data.X_test, data.Y_test)
    else:
        raise RuntimeError("Unknown dataset type detected")

    (x_train, x_mu, x_std), (y_train, y_mu, y_std) = norm_dataset(train)
    x_test = (test[0] - x_mu) / x_std
    y_test = (test[1] - y_mu) / y_std
    return DatasetBundle(name, _to_dtype(x_train, y_train), _to_dtype(x_test, y_test))

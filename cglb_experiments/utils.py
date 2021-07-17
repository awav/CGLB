from typing import Tuple
from pathlib import Path


def short_names(results_path: Path) -> Tuple[str, str]:
    dataset, name = str(results_path.parent).split("/")[-2:]
    d = create_short_dataset_name(dataset)
    n = create_short_name(name)
    return d, n


def create_short_name(name: str) -> str:
    parts = name.split("-")
    model = parts[0]
    kernel = "SE" if parts[1] == "SquaredExponential" else parts[1]

    short_name = f"{model}-{kernel}"

    ip_prefix = "num_iv_"
    ip = None
    for p in parts:
        if p.startswith(ip_prefix):
            ip = p.replace(ip_prefix, "")
            break

    if ip is None:
        return short_name

    return f"{short_name}-{ip}"


def create_short_dataset_name(name: str) -> str:
    return name.replace("Wilson_", "")

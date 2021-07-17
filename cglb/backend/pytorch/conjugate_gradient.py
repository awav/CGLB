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

from dataclasses import dataclass
from typing import Callable, Tuple, Union, Optional
import torch
from pykeops.torch import LazyTensor
from gpytorch.lazy import LazyTensor as GPytorchLazyTensor

Tensor = torch.Tensor
Preconditioner = Callable[[Tensor], Tuple[Tensor, Tensor]]


@dataclass
class ConjugateGradientStats:
    steps: Union[Tensor, float]
    residual_error: Union[Tensor, float]


@dataclass
class ConjugateGradient:
    """
    CG stops if: 0.5 * rᵀQ⁻¹r < max_error || i > max_cg_iter
    """

    max_error: float = 1.0
    max_cg_iter: int = 100
    restart_cg_iter: int = 40

    def __call__(
        self,
        A: Union[Tensor, LazyTensor, GPytorchLazyTensor],
        b: Tensor,
        v: Tensor,
        precond: Preconditioner,
    ) -> Tuple[Tensor, ConjugateGradientStats]:
        """
        :param A: [N, N] Matrix we want to backsolve from
        :param b: [B, N] Vector we want to backsolve
        :param P: Preconditioner (implements a mat_vec method)
        :return: v: Approximate solution
        """

        v = v.clone()
        # CG initialization
        Av = A @ v
        r = b - Av
        z, rz = precond(r)
        # rz = rᵀPr, where z = Pr
        p = z

        i: int = 0
        # CG loop
        while (0.5 * rz > self.max_error) and (i < self.max_cg_iter):
            Ap = A @ p
            gamma = rz / (p * Ap).sum()
            v += gamma * p

            restart = i % self.restart_cg_iter == self.restart_cg_iter - 1

            r = (b - A @ v) if restart else (r - gamma * Ap)
            z, new_rz = precond(r)

            p = z if restart else (z + p * new_rz / rz)
            rz = new_rz
            i += 1

            # del Ap
            if p.is_cuda:
                torch.cuda.synchronize()

        stats = ConjugateGradientStats(i, 0.5 * rz.detach().cpu())

        # del Av, p, r, rz
        return v, stats


@dataclass
class NystromPreconditioner:
    A: Tensor  # [N, N]
    LB: Tensor  # [M, M]
    sigma_sq: Tensor  # []

    def __call__(self, r: Tensor) -> Tuple[Tensor, Tensor]:  # ([N, 1], [])
        """
        Args:
            r: Vector [N, 1]
        Return:
            A tuple of a vector [N, 1] and a scalar []
        """
        A, LB, sigma_sq = (self.A, self.LB, self.sigma_sq)
        trisolve = torch.triangular_solve

        Ar = A @ r
        LBinvAr = trisolve(Ar, LB, upper=False).solution
        LBinvtLBinvAr = trisolve(LBinvAr, LB.transpose(-1, -2)).solution

        # TODO: better to do - `A.transpose(-1, -2) @ LBinvtLBinvAr`?
        p = LBinvtLBinvAr.transpose(-1, -2) @ A
        rp = r - p.transpose(-1, -2)
        rpr = (rp * r).sum()
        return rp / sigma_sq, rpr / sigma_sq
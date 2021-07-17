from gpflow.monitor.tensorboard import ToTensorBoard
from cglb.backend.pytorch.conjugate_gradient import ConjugateGradient
from dataclasses import dataclass
from typing import Tuple, Union, Optional

import gpytorch
import numpy as np
import torch

# from GPUtil import showUtilization as gpu_usage
# import torch.autograd.profiler as profiler

from torch import Tensor
from .conjugate_gradient import ConjugateGradient, NystromPreconditioner, ConjugateGradientStats
from gpytorch import settings, delazify
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.lazy import LazyTensor as GPytorchLazyTensor


Tensor = torch.Tensor
GenericTensor = Union[np.ndarray, Tensor]
Data = Tuple[GenericTensor, GenericTensor]


class GPR(gpytorch.models.ExactGP):
    def __init__(self, data: Data, likelihood, kernel):
        super().__init__(data[0], data[1], likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SGPR(GPR):
    ...


class CGLB(SGPR):
    def __init__(self, *args, **kwargs):
        super(CGLB, self).__init__(*args, **kwargs)
        self._v_vec = self._build_v_vec()

    def _build_v_vec(self) -> torch.Tensor:
        num_data = self.train_targets.shape[0]
        target_dim = 1  # NOTE: we tackle only one dimensional target.
        v_vec_shape = (int(num_data), target_dim)
        return torch.zeros(
            v_vec_shape,
            dtype=self.train_targets.dtype,
            device=self.train_targets.device,
            requires_grad=False,
        )

    @property
    def v_vec(self) -> Tensor:
        return self._v_vec

    @property
    def cg_stats(self) -> Optional[ConjugateGradientStats]:
        if hasattr(self, "_cg_stats"):
            return self._cg_stats
        return None

    @cg_stats.setter
    def cg_stats(self, value: ConjugateGradientStats):
        steps, error = value.steps, value.residual_error
        if isinstance(value.steps, torch.Tensor):
            steps = value.steps.detach().cpu().numpy()
        if isinstance(value.residual_error, torch.Tensor):
            error = value.residual_error.detach().cpu().numpy()
        self._cg_stats = ConjugateGradientStats(steps, error)


@dataclass
class CommonTerms:
    A: Tensor
    LB: Tensor
    AAt_diag_sum: Tensor
    L: Tensor


@dataclass
class Bounds:
    upper_bound: Tensor
    lower_bound: Tensor


class LowerBoundCG(ExactMarginalLogLikelihood):
    def __init__(
        self,
        model: SGPR,
        cg_opt: Optional[ConjugateGradient] = None,
        use_cache: bool = False,
        cached_v_vec_initial: bool = False,
    ):
        if not isinstance(model, SGPR):
            raise ValueError(f"CGLB model expected in the constructor of the {self.__class__}")

        super().__init__(model.likelihood, model)
        targets = model.train_targets
        self.cg_opt = ConjugateGradient() if cg_opt is None else cg_opt
        self._cached_v_vec = cached_v_vec_initial

        # TODO: do not use cached vector
        self._use_cache = use_cache

    @property
    def cached_v_vec(self) -> bool:
        return self._cached_v_vec

    @cached_v_vec.setter
    def cached_v_vec(self, value: bool):
        self._cached_v_vec = value

    @property
    def mean(self) -> gpytorch.means.Mean:
        return self.model.mean_module

    @property
    def likelihood(self) -> gpytorch.likelihoods.Likelihood:
        return self.model.likelihood

    @property
    def kernel(self) -> gpytorch.kernels.Kernel:
        return self.model.covar_module.base_kernel

    @property
    def inducing_points(self) -> Tensor:
        return self.model.covar_module.inducing_points

    @property
    def noise(self) -> Tensor:
        return self.likelihood.noise.squeeze()

    def forward(self, data: Tuple[Tensor, Tensor], *params):

        # with profiler.profile(with_stack=True, profile_memory=True) as prof:

        terms = self.logdet_and_quad_common_terms(data)
        _inputs, targets = data
        dtype = terms.A.dtype

        num_data, output_dim = _output_dims(targets)

        # Constant term
        pi2 = torch.log(torch.as_tensor(2.0 * np.pi, dtype=dtype))
        const_term = -0.5 * num_data * output_dim * pi2
        # Log det term
        logdet_term = self.logdet_estimator(data, terms)

        # Estimate of Quadratic form from CGQ
        quad_terms = self.quad_estimator(data, terms)
        bound = quad_terms.upper_bound + logdet_term + const_term

        # print(prof.key_averages().table(sort_by="cuda_memory_usage"))
        # prof.export_chrome_trace("/tmp/cglb.prof")

        return bound

    def logdet_and_quad_common_terms(
        self,
        data: Tuple,
    ) -> CommonTerms:
        """
        Matrices used in log-det calculation/preconditioner
        :return: A, LB, AAT, all square NxN matrices
        """
        kernel = self.kernel
        inducing_variable = self.inducing_points

        x_data, y_data = data
        sigma_sq = self.noise

        num_inducing = len(inducing_variable)
        sigma = torch.sqrt(sigma_sq)
        jitter = settings.cholesky_jitter.value()
        jitter = torch.tensor(jitter, dtype=sigma.dtype, device=sigma.device)

        # kuf_lazy: GPytorchLazyTensor = _eval_kernel(kernel, inducing_variable, x_data)
        kuf_lazy: GPytorchLazyTensor = kernel(inducing_variable, x_data)
        kuf: Tensor = delazify(kuf_lazy)

        # kuu: GPytorchLazyTensor = _eval_kernel(kernel, inducing_variable, inducing_variable)
        kuu: GPytorchLazyTensor = kernel(inducing_variable, inducing_variable)
        kuu_jitter: Tensor = delazify(kuu.add_diag(jitter))
        kuu_chol = torch.cholesky(kuu_jitter)

        # Compute intermediate matrices
        trisolve = torch.triangular_solve
        A = trisolve(kuf, kuu_chol, upper=False).solution / sigma
        AAt = A @ A.transpose(-1, -2)
        I = torch.eye(num_inducing, dtype=AAt.dtype, device=AAt.device)
        B = AAt + I
        LB = torch.cholesky(B)
        AAt_diag_sum = AAt.diagonal().sum()

        return CommonTerms(A=A, LB=LB, AAt_diag_sum=AAt_diag_sum, L=kuu_chol)

    def logdet_estimator(
        self,
        data: Tuple,
        terms: CommonTerms,
    ) -> Tensor:
        """
        Bound from Jensen's Inequality:
        log |K + σ²I| <= log |Q + σ²I| + N * log(1 + tr(K - Q)/(σ²N))

        :param terms: Common terms. LB, and AAt is used.
        :return: log_det, upper bound on .5 * log|K + σ²I|
        """

        x_data, y_data = data
        num_data, output_dim = _output_dims(y_data)
        kernel = self.kernel

        sigma_sq = self.noise
        kdiag = kernel(x_data, diag=True)

        # t / σ²
        trace = kdiag.sum() / sigma_sq - terms.AAt_diag_sum

        # log|Q + σ²I|
        logdet = -output_dim * terms.LB.diagonal().log().sum()
        logdet -= 0.5 * num_data * output_dim * torch.log(sigma_sq)

        # Correction term from Jensen's inequality
        logdet -= 0.5 * output_dim * num_data * torch.log(1.0 + trace / num_data)
        return logdet

    def quad_estimator(self, data, terms) -> Bounds:
        sigma_sq = self.noise
        kernel = self.kernel

        x_data, y_data = data
        kxx: GPytorchLazyTensor = kernel(x_data)
        cov: GPytorchLazyTensor = kxx.add_diag(sigma_sq)
        mean: Tensor = self.mean(x_data)
        err = y_data.reshape(-1, 1) - mean.reshape(-1, 1)

        # Get upper and lower bounds on .5 * yTK^{-1}y
        cov_detached = cov.detach()
        err_detached = err.detach()

        precon = NystromPreconditioner(terms.A, terms.LB, sigma_sq)

        with torch.no_grad():
            if self._use_cache and self.cached_v_vec:
                v = self.model.v_vec
            else:
                v, cg_stats = self.cg_opt(
                    cov_detached,
                    err_detached,
                    self.model.v_vec,
                    precon,
                )
                self.model.cg_stats = cg_stats
                # Use previous solution to initialize next run of CG
                self.model.v_vec.data.copy_(v.data)

                # L-BFGS-B runs a line search. We should re-use `v` vector during the line search.
                # Flag helps to prevent unnecessary CG runs.
                self.cached_v_vec = self._use_cache

        cov_v = cov @ v
        r = err - cov_v
        _, error_bound = precon(r)
        lower_bound = (v * (r + 0.5 * cov_v)).sum()
        upper_bound = lower_bound + 0.5 * error_bound

        return Bounds(upper_bound=-upper_bound, lower_bound=-lower_bound)


class PredictCG(LowerBoundCG):
    def __init__(self, model: SGPR, cg_opt: Optional[ConjugateGradient] = None):
        cg_opt = ConjugateGradient(max_error=1e-3) if cg_opt is None else cg_opt
        super().__init__(model, cg_opt)
        # Detach before cloning in case v requires grad ever
        self._v_vec = model.v_vec.detach().clone()
        self.cached = False
        self.terms = None

    @property
    def v_vec(self):
        return self._v_vec

    def clear_cache(self):
        self.v_vec.copy_(self.model.v_vec.detach().clone())
        self.cached = False
        self.terms = None

    def forward(
        self, xnew: Tensor, full_cov: bool = False, full_output_cov: bool = False
    ) -> Tuple[Tensor, Tensor]:

        if full_cov:
            raise NotImplementedError(
                "The predict_f method currently  supports only `full_cov=False` option"
            )

        x, *_ = self.model.train_inputs
        y = self.model.train_targets.reshape(-1, 1)
        err = y - self.mean(x).reshape(-1, 1)
        sigma_sq = self.noise
        ksf = self.kernel(xnew, x)
        cov = self.kernel(x).add_diag(sigma_sq)

        if self.cached:
            terms = self.terms
            new_v = self.v_vec
        else:
            terms = self.logdet_and_quad_common_terms((x, y))
            precon = NystromPreconditioner(terms.A, terms.LB, sigma_sq)
            new_v, cg_stats = self.cg_opt(cov, err, self.v_vec, precon)
            self.v_vec.data.copy_(new_v)
            self.terms = terms
            self.cached = True

        cg_mean = ksf @ new_v
        res = err - cov @ new_v

        kus = self.kernel(self.inducing_points, xnew)
        sigma = torch.sqrt(sigma_sq)

        a_res = terms.A @ res

        trisolve = torch.triangular_solve
        c = trisolve(a_res, terms.LB, upper=False)[0] / sigma
        tmp1 = trisolve(delazify(kus), terms.L, upper=False)[0]
        tmp2 = trisolve(tmp1, terms.LB, upper=False)[0]

        sgpr_mean = tmp2.transpose(-1, -2) @ c
        f_mean = sgpr_mean + cg_mean + self.mean(xnew).reshape(-1, 1)

        kss = self.kernel(xnew, diag=True)
        f_var = delazify(kss) + (tmp2 ** 2).sum(0) - (tmp1 ** 2).sum(0)
        f_var = f_var.reshape(*f_mean.shape)

        return f_mean, f_var


class PredictLogdensityCG(LowerBoundCG):
    def forward(
        self, data: Tuple[Tensor, Tensor], full_cov: bool = False, full_output_cov: bool = False
    ):
        if full_cov or full_output_cov:
            raise NotImplementedError(
                "The predict_log_density method currently supports only the argument values full_cov=False and full_output_cov=False"
            )
        x, y = data
        f_mean, f_var = super().forward(x, full_cov=full_cov, full_output_cov=full_output_cov)
        return gaussian(y, f_mean, f_var + self.noise).sum(axis=-1)


def log_density(m, y, f_mean, f_var) -> Tensor:
    noise = m.likelihood.noise.squeeze()
    return gaussian(y, f_mean, f_var + noise).sum(axis=-1)


def gaussian(x, mu, var):
    pi = torch.tensor(np.pi, dtype=x.dtype, device=x.device)
    pi2 = torch.log(2 * pi)
    x = x.reshape(*mu.shape)
    return -0.5 * (pi2 + torch.log(var) + (mu - x) ** 2 / var)


def _output_dims(t: Tensor) -> Tuple[Tensor, Tensor]:
    num_data = torch.tensor(t.size(0), dtype=t.dtype)
    output_dim = torch.tensor(t.size(1) if t.ndim == 2 else 1, dtype=t.dtype)
    return num_data, output_dim

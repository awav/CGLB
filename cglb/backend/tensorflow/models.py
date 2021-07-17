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

from typing import Tuple
import tensorflow as tf
import numpy as np
from gpflow import Parameter
from gpflow.utilities import to_default_float
from gpflow.covariances.dispatch import Kuf, Kuu
from gpflow.config import default_float, default_int, default_jitter
from gpflow.models import SGPR
from gpflow.models.training_mixins import InputData, RegressionData

from .preconditioners import NystromPreconditioner
import collections

MeanAndVariance = Tuple[tf.Tensor, tf.Tensor]


class CGLB(SGPR):
    def __init__(
        self,
        data,
        *args,
        max_error: float = 1.0,
        max_cg_iters=100,
        restart_cg_iters=40,
        joint_optimization=False,
        vzero: bool = False,
        **kwargs,
    ):
        super(CGLB, self).__init__(data, *args, **kwargs)
        # Vector used to initialize CG. Updated to solution of last run
        N, B = self.data[1].shape
        v0_trainable = joint_optimization and not vzero
        self.v0 = Parameter(tf.zeros((B, N), dtype=default_float()), trainable=v0_trainable)
        # CG stops if: .5 * rᵀQ⁻¹r < max_error, or i > self.max_cg_iters
        self.max_error = tf.Variable(max_error, trainable=False, dtype=default_float())
        self.max_cg_iters = max_cg_iters
        self.restart_cg_iters = restart_cg_iters
        self.joint_optimization = joint_optimization
        self.vzero = vzero

        self.cg_steps = tf.Variable(0, trainable=False)
        self.cg_residual_error = tf.Variable(0.0, trainable=False, dtype=default_float())

    def _common_calculation(self):
        """
        Matrices used in log-det calculation/preconditioner
        :return: A, B, LB, AAT all square NxN matrices
        """
        CommonTensors = collections.namedtuple("CommonTensors", ["A", "B", "LB", "AAT", "L"])
        X_data, Y_data = self.data
        num_inducing = tf.shape(self.inducing_variable.Z)[0]
        kuf = Kuf(self.inducing_variable, self.kernel, X_data)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        L = tf.linalg.cholesky(kuu)
        sigma = tf.sqrt(self.likelihood.variance)
        # Compute intermediate matrices
        A = tf.linalg.triangular_solve(L, kuf, lower=True) / sigma
        AAT = tf.linalg.matmul(A, A, transpose_b=True)
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        return CommonTensors(A, B, LB, AAT, L)

    def _log_det_estimator(self, common):
        """
        Bound from Jensen's Inequality:
        log |K + σ²I| <= log |Q + σ²I| + N * log (1 + tr(K - Q)/(σ²N))

        :param B: Matrix returned from common calc, unused
        :param LB: Matrix returned from common calc
        :param AAT: Matrix returned from common calc
        :return: log_det, upper bound on .5 * log |K + σ²I|
        """
        LB = common.LB
        AAT = common.AAT
        X_data, Y_data = self.data
        num_data, output_dim = (
            to_default_float(tf.shape(Y_data)[0]),
            to_default_float(tf.shape(Y_data)[1]),
        )
        Kdiag = self.kernel(X_data, full_cov=False)
        # t / σ²
        trace = tf.reduce_sum(Kdiag) / self.likelihood.variance - tf.reduce_sum(
            tf.linalg.diag_part(AAT)
        )
        # log|Q + σ²I|
        log_det = -output_dim * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))
        log_det -= 0.5 * num_data * output_dim * tf.math.log(self.likelihood.variance)

        # Correction term from Jensen's inequality
        log_det -= 0.5 * output_dim * num_data * tf.math.log(1 + trace / num_data)
        return log_det

    def _preconditioned_cgq(self, A, b, P, max_error, max_iters):
        """
        :param A: [N, N] Matrix we want to backsolve from
        :param b: [B, N] Vector we want to backsolve
        :param P: Preconditioner (implements a mat_vec method)
        :return: v: Approximate solution
        """
        CGState = collections.namedtuple("CGState", ["i", "v", "r", "p", "rz"])

        def stopping_criterion(state):
            return (0.5 * state.rz > max_error) and (state.i < max_iters)

        def cg_step(state):
            Ap = tf.matmul(state.p, A)
            denom = tf.reduce_sum(state.p * Ap, axis=-1)
            gamma = state.rz / denom
            v = state.v + gamma * state.p
            i = state.i + 1
            r = tf.cond(
                state.i % self.restart_cg_iters == self.restart_cg_iters - 1,
                lambda: b - tf.matmul(v, A),
                lambda: state.r - gamma * Ap,
            )
            z, new_rz = P.mat_vec(r)
            p = tf.cond(
                state.i % self.restart_cg_iters == self.restart_cg_iters - 1,
                lambda: z,
                lambda: z + state.p * new_rz / state.rz,
            )
            return [CGState(i, v, r, p, new_rz)]

        Av = tf.matmul(self.v0, A)
        r = b - Av
        z, rz = P.mat_vec(r)
        p = z
        i = tf.constant(0, dtype=default_int())
        initial_state = CGState(i, self.v0, r, p, rz)
        final_state = tf.while_loop(stopping_criterion, cg_step, [initial_state])
        final_state = tf.nest.map_structure(tf.stop_gradient, final_state)
        self.cg_steps.assign(final_state[0].i)
        self.cg_residual_error.assign(0.5 * final_state[0].rz)
        return final_state[0].v

    def _CGQ_quad_form(self, A, LB):
        X_data, Y_data = self.data
        err = Y_data - self.mean_function(X_data)
        num_data = to_default_float(tf.shape(err)[0])
        K = self.kernel.K(X_data) + self.likelihood.variance * tf.eye(
            num_data, dtype=default_float()
        )
        # For now, we always use preconditioner
        P = NystromPreconditioner(A, LB, self.likelihood.variance)
        # Get upper and lower bounds on 0.5 * yᵀK⁻¹y
        err_t = tf.transpose(err)
        if self.joint_optimization or self.vzero:
            v = self.v0
        else:
            v = self._preconditioned_cgq(K, err_t, P, self.max_error, self.max_cg_iters)

        Kv = tf.matmul(v, K)
        r = err_t - Kv
        _, error_bound = P.mat_vec(r)
        lb = tf.reduce_sum(v * (r + 0.5 * Kv))
        ub = lb + 0.5 * error_bound
        if not (self.joint_optimization or self.vzero):
            self.v0.assign(v)
        return -ub

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        """
        Compute a lower bound on log marginal likelihood using CG for quadratic term and Nystrom + Jensen inequality
        for log det
        :return: bound <= L(y)
        """

        X_data, Y_data = self.data
        num_data = to_default_float(tf.shape(Y_data)[0])
        output_dim = to_default_float(tf.shape(Y_data)[1])
        common = self._common_calculation()
        # Constant term
        bound = -0.5 * num_data * output_dim * np.log(2 * np.pi)
        # Log det term
        bound += self._log_det_estimator(common)
        # Estimate of Quadratic form from CGQ
        bound += self._CGQ_quad_form(common.A, common.LB)
        return bound

    def predict_f(
        self, Xnew: InputData, full_cov=False, full_output_cov=False, cg_tolerance=1e-3
    ) -> MeanAndVariance:
        """
        Modified version of SGPR in GPflow:
        The posterior mean for our model is given by m(xs) = Ksf*v + Qff(Qff+sigma^I)^{-1} r
        where r = y-(Kff+ sigma^2 I) v is the residual from CG. Note that when v=0, this agrees
        with the SGPR mean, while if v = (Kff+sigma^2 I)^{-1}y, r=0, and the exact GP mean is
        recovered.
        """
        X_data, Y_data = self.data
        err = Y_data - self.mean_function(X_data)
        kxx = self.kernel(X_data, X_data)
        ksf = self.kernel(Xnew, X_data)
        eye_sigma = self.likelihood.variance * tf.eye(kxx.shape[0], dtype=default_float())
        kmat = kxx + eye_sigma
        CommonTensors = self._common_calculation()
        A, LB, L = CommonTensors.A, CommonTensors.LB, CommonTensors.L
        if cg_tolerance is None or self.joint_optimization or self.vzero:
            v = self.v0
        else:
            P = NystromPreconditioner(A, LB, self.likelihood.variance)
            old_cg_steps = self.cg_steps
            old_resid_error = self.cg_residual_error
            err_t = tf.transpose(err)
            v = self._preconditioned_cgq(kmat, err_t, P, cg_tolerance, self.max_cg_iters)
            self.cg_steps.assign(old_cg_steps)
            self.cg_residual_error.assign(old_resid_error)
        cg_mean = tf.matmul(ksf, v, transpose_b=True)
        res = err - tf.matmul(kmat, v, transpose_b=True)

        Kus = Kuf(self.inducing_variable, self.kernel, Xnew)
        sigma = tf.sqrt(self.likelihood.variance)
        Ares = tf.linalg.matmul(A, res)
        c = tf.linalg.triangular_solve(LB, Ares, lower=True) / sigma
        tmp1 = tf.linalg.triangular_solve(L, Kus, lower=True)
        tmp2 = tf.linalg.triangular_solve(LB, tmp1, lower=True)
        sgpr_mean = tf.linalg.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = (
                self.kernel(Xnew)
                + tf.linalg.matmul(tmp2, tmp2, transpose_a=True)
                - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
            )
            var = tf.tile(var[None, ...], [self.num_latent_gps, 1, 1])  # [P, N, N]
        else:
            var = (
                self.kernel(Xnew, full_cov=False)
                + tf.reduce_sum(tf.square(tmp2), 0)
                - tf.reduce_sum(tf.square(tmp1), 0)
            )
            var = tf.tile(var[:, None], [1, self.num_latent_gps])
        return sgpr_mean + cg_mean + self.mean_function(Xnew), var

    def predict_log_density(
        self,
        data: RegressionData,
        full_cov: bool = False,
        full_output_cov: bool = False,
        cg_tolerance=1e-6,
    ) -> tf.Tensor:
        """
        Compute the log density of the data at the new data points.
        """
        if full_cov or full_output_cov:
            raise NotImplementedError(
                "The predict_log_density method currently supports only the argument values full_cov=False and full_output_cov=False"
            )

        X, Y = data
        f_mean, f_var = self.predict_f(
            X, full_cov=full_cov, full_output_cov=full_output_cov, cg_tolerance=cg_tolerance
        )
        return self.likelihood.predict_log_density(f_mean, f_var, Y)


class CGLBNM2(CGLB):
    def _log_det_estimator(self, common):
        """
        NM^2 bound on log determinant `\log\det(Q) + tr(K - Q) / \sigma^2`.

        :param common: Common calculation
        :return: log_det, upper bound on `0.5 * log |K + σ²I|`
        """
        LB = common.LB
        AAT = common.AAT
        A = common.A

        trisolve = tf.linalg.triangular_solve
        trace = tf.linalg.trace
        rsum = tf.reduce_sum
        diag_part = tf.linalg.diag_part
        matmul = tf.linalg.matmul
        log = tf.math.log

        X_data, Y_data = self.data
        n, output_dim = (
            to_default_float(tf.shape(Y_data)[0]),
            to_default_float(tf.shape(Y_data)[1]),
        )

        sigma_2 = self.likelihood.variance

        # Trace part `tr(K - Q) / \sigma^2)`
        # where `C = L_b^{-1} A`

        kdiag = self.kernel(X_data, full_cov=False)
        trace = tf.reduce_sum(kdiag) / self.likelihood.variance - tf.reduce_sum(
            tf.linalg.diag_part(AAT)
        )

        # \log \det (Q_{ff} + σ²I)
        log_det_q = rsum(log(diag_part(LB))) + 0.5 * n * log(sigma_2)

        return -(log_det_q + 0.5 * trace)

class CGLBN2M(CGLB):
    def _log_det_estimator(self, common):
        """
        N^2M bound on log determinant `\log\det(Q) + n \log(tr(Q^{-1}K) / n)`.

        :param common: Common calculation
        :return: log_det, upper bound on `0.5 * log |K + σ²I|`
        """
        LB = common.LB
        AAT = common.AAT
        A = common.A

        trisolve = tf.linalg.triangular_solve
        trace = tf.linalg.trace
        rsum = tf.reduce_sum
        diag_part = tf.linalg.diag_part
        matmul = tf.linalg.matmul
        log = tf.math.log

        X_data, Y_data = self.data
        n, output_dim = (
            to_default_float(tf.shape(Y_data)[0]),
            to_default_float(tf.shape(Y_data)[1]),
        )

        sigma_2 = self.likelihood.variance
        kff = self.kernel(X_data)
        kff_sigma_2 = _add_scalar_to_diag(kff, sigma_2)

        # Log-trace part
        # `n \log (tr(Q^{-1}K) / n) = n \log (tr(K) - tr(CKC^{T})) - n \log (n sigma^2)`
        # where `C = L_b^{-1} A`
        trace_kff = trace(kff_sigma_2)
        C = trisolve(LB, A)
        trace_qrest = trace(matmul(C @ kff_sigma_2, C, transpose_b=True))
        log_trace = n * (log(trace_kff - trace_qrest) - log(n) - log(sigma_2))

        # \log \det (Q_{ff} + σ²I)
        log_det_q = rsum(log(diag_part(LB))) + 0.5 * n * log(sigma_2)

        return -(log_det_q + 0.5 * log_trace)


class SGPRN2M(SGPR):
    def elbo(self) -> tf.Tensor:
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. For a derivation of the terms in here, see the associated
        SGPR notebook.
        """
        X_data, Y_data = self.data

        trisolve = tf.linalg.triangular_solve
        trace = tf.linalg.trace
        rsum = tf.reduce_sum
        diag_part = tf.linalg.diag_part
        matmul = tf.linalg.matmul
        log = tf.math.log

        num_inducing = self.inducing_variable.num_inducing
        num_data = to_default_float(tf.shape(Y_data)[0])
        output_dim = to_default_float(tf.shape(Y_data)[1])

        sigma_2 = self.likelihood.variance
        n = num_data

        err = Y_data - self.mean_function(X_data)
        kff = self.kernel(X_data, full_cov=True)
        kff_sigma_2 = _add_scalar_to_diag(kff, sigma_2)
        kuf = Kuf(self.inducing_variable, self.kernel, X_data)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        L = tf.linalg.cholesky(kuu)
        sigma = tf.sqrt(self.likelihood.variance)

        # Compute intermediate matrices
        A = tf.linalg.triangular_solve(L, kuf, lower=True) / sigma
        AAT = tf.linalg.matmul(A, A, transpose_b=True)
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        Aerr = tf.linalg.matmul(A, err)
        c = tf.linalg.triangular_solve(LB, Aerr, lower=True) / sigma

        # compute log marginal bound
        bound = -0.5 * num_data * output_dim * np.log(2 * np.pi)
        bound += tf.negative(output_dim) * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))
        bound -= 0.5 * num_data * output_dim * tf.math.log(self.likelihood.variance)
        bound += -0.5 * tf.reduce_sum(tf.square(err)) / self.likelihood.variance
        bound += 0.5 * tf.reduce_sum(tf.square(c))

        # Log-trace part
        # `n \log (tr(Q^{-1}K) / n) = n \log (tr(K) - tr(CKC^{T})) - n \log (n sigma^2)`
        # where `C = L_b^{-1} A`

        trace_kff = trace(kff_sigma_2)
        C = trisolve(LB, A)
        trace_qrest = trace(matmul(C @ kff_sigma_2, C, transpose_b=True))
        log_trace = n * (log(trace_kff - trace_qrest) - log(n) - log(sigma_2))

        # bound += -0.5 * output_dim * tf.reduce_sum(Kdiag) / self.likelihood.variance
        # bound += 0.5 * output_dim * tf.reduce_sum(tf.linalg.diag_part(AAT))

        bound -= 0.5 * log_trace

        return bound


def _add_scalar_to_diag(K, v):
    diag = tf.linalg.diag_part(K)
    diag = diag + v
    return tf.linalg.set_diag(K, diag)


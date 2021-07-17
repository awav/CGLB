import tensorflow as tf
import abc


class Preconditioner:
    @abc.abstractmethod
    def mat_vec(self, v):
        raise NotImplementedError

    @abc.abstractmethod
    def inv_mat_vec(self, v):
        raise NotImplementedError


class Eye(Preconditioner):
    def mat_vec(self, v):
        return v

    def inv_mat_vec(self, v):
        return v


class NystromPreconditioner(Preconditioner):
    """
    Preconditioner of the form \math:`(Qff + σ²I)⁻¹`, where \math:`A = σ⁻²L⁻¹Kᵤₓ` and \math:`B = AAᵀ + I`
    """

    def __init__(self, A, LB, lam):
        super(NystromPreconditioner, self).__init__()
        self.A = A
        self.LB = LB
        self.lam = lam

    def mat_vec(self, v):
        """
        (KufL^TLKuf + \lambda I)^{-1} v = v / lambda - 1/lambda * Kuf^T(
        :param v:
        :return:
        """
        sigma_sq = self.lam
        A = self.A
        LB = self.LB

        trans = tf.transpose
        trisolve = tf.linalg.triangular_solve
        matmul = tf.linalg.matmul
        sum = tf.reduce_sum
        square = tf.math.square

        v = trans(v)
        Av = matmul(A, v)
        LBinvAv = trisolve(LB, Av)
        LBinvtLBinvAv = trisolve(trans(LB), LBinvAv, lower=False)

        rv = v - matmul(A, LBinvtLBinvAv, transpose_a=True)

        # ATTN: which approach brings cholesky errors? does it matter?
        # vtrv = sum(square(v)) - sum(square(LBinvAv))
        vtrv = sum(rv * v)

        # ATTN: difference between two approaches
        # tf.debugging.assert_equal(sum(rv * v) / sigma_sq, vtrv)

        return trans(rv) / sigma_sq, vtrv / sigma_sq

    def inv_mat_vec(self, v):
        v = tf.transpose(v)
        v_lam = v * self.lam
        Av = tf.matmul(self.A, v_lam)
        Qffv = tf.matmul(self.A, Av, transpose_a=True)
        return Qffv + v_lam

    def inv_sqrt_mat_vec(self, v):
        v_lam = v * tf.sqrt(self.lam)
        vA = tf.matmul(v_lam, self.A, transpose_b=True)
        return tf.transpose(vA + v_lam)

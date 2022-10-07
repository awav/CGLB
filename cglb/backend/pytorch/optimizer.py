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

import numpy as np
import scipy
import torch


class Scipy:
    def minimize(self, closure, variables, method="L-BFGS-B", step_callback=None, **scipy_kwargs):
        variables = tuple(variables)
        init_vals = self.to_numpy(self.pack(variables))
        func = self.eval_func(closure, variables)
        if step_callback is not None:
            callback = self.callback_func(variables, step_callback)
            scipy_kwargs.update(dict(callback=callback))
        return scipy.optimize.minimize(func, init_vals, jac=True, method=method, **scipy_kwargs)

    @classmethod
    def eval_func(cls, closure, variables):
        device = variables[0].device

        def _torch_eval(x):
            values = cls.unpack(variables, x)
            cls.assign(variables, values)

            loss, grads = _compute_loss_and_gradients(closure, variables)
            return loss, cls.pack(grads)

        def _eval(x):
            loss, grad = _torch_eval(torch.from_numpy(x).to(device))
            return (
                loss.cpu().detach().numpy().astype(np.float64),
                grad.cpu().detach().numpy().astype(np.float64),
            )

        return _eval

    @classmethod
    def callback_func(cls, variables, step_callback):
        step = 0

        def _callback(x):
            nonlocal step
            device = variables[0].device
            values = cls.unpack(variables, torch.from_numpy(x).to(device))
            step_callback(step, variables, values)
            step += 1

        return _callback

    @staticmethod
    def pack(tensors):
        flats = [torch.flatten(tensor) for tensor in tensors]
        tensors_vector = torch.cat(flats, axis=0)
        return tensors_vector

    @staticmethod
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy()

    @staticmethod
    def unpack(to_tensors, from_vector):
        s = 0
        values = []
        for target_tensor in to_tensors:
            shape = torch.tensor(target_tensor.shape)
            dtype = target_tensor.dtype
            tensor_size = int(torch.prod(shape))
            tensor_vector = from_vector[s : s + tensor_size]
            tensor = torch.reshape(tensor_vector.type(dtype), tuple(shape))
            values.append(tensor)
            s += tensor_size
        return values

    @staticmethod
    def assign(to_tensors, values):
        if len(to_tensors) != len(values):
            raise ValueError("to_tensors and values should have same length")
        for target, value in zip(to_tensors, values):
            target.data = value


def _compute_loss_and_gradients(loss_closure, variables):
    loss = loss_closure()
    grads = torch.autograd.grad(loss, variables)
    return loss, grads
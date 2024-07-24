import numpy as np
from .function import Function
from .value import Value
import pursuitnet as pn

class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return pn.Tensor(np.add(a.data, b.data), requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = pn.Tensor(grad_output.data) if a.requires_grad else None
        grad_b = pn.Tensor(grad_output.data) if b.requires_grad else None
        return grad_a, grad_b

class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Value(a.data * b.data, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_output * b.data if a.requires_grad else None
        grad_b = grad_output * a.data if b.requires_grad else None
        return grad_a, grad_b

class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Value(np.matmul(a.data, b.data), requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = np.matmul(grad_output, b.data.T) if a.requires_grad else None
        grad_b = np.matmul(a.data.T, grad_output) if b.requires_grad else None
        return grad_a, grad_b

class Sum(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return Value(np.sum(input.data), requires_grad=input.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return np.full_like(input.data, grad_output)
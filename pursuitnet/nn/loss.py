import numpy as np
from .module import Module
import pursuitnet as pn
from pursuitnet.autograd.function import Function

class CrossEntropyLossFunction(Function):
    @staticmethod
    def forward(ctx, input, target):
        input_data = input.data
        target_data = target.data.astype(int)
        
        assert len(input_data.shape) == 2, "Input must be a 2D array"
        assert len(target_data.shape) == 1, "Target must be a 1D array"
        
        exp_input = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        softmax_output = exp_input / np.sum(exp_input, axis=1, keepdims=True)
        
        m = target_data.shape[0]
        log_likelihood = -np.log(softmax_output[np.arange(m), target_data])
        loss = np.sum(log_likelihood) / m
        
        ctx.save_for_backward(input, target, softmax_output)
        return pn.Tensor(np.array([loss]), requires_grad=input.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        input, target, softmax_output = ctx.saved_tensors
        m = target.shape[0]
        
        grad_input = softmax_output.copy()
        grad_input[np.arange(m), target.data.astype(int)] -= 1
        grad_input /= m
        
        if grad_output is not None:
            grad_input *= grad_output.data

        return pn.Tensor(grad_input, requires_grad=input.requires_grad), None

class CrossEntropyLoss(Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, input, target):
        return CrossEntropyLossFunction.apply(input, target)

    def __call__(self, input, target):
        return self.forward(input, target)
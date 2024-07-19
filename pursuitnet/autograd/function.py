from .value import Value

class Function:
    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

    @classmethod
    def apply(cls, *args, **kwargs):
        ctx = Context()
        result = cls.forward(ctx, *args, **kwargs)
        if any(arg.requires_grad for arg in args if isinstance(arg, Value)):
            def backward_function(grad_output):
                grads = cls.backward(ctx, grad_output)
                for arg, grad in zip(args, grads):
                    if isinstance(arg, Value) and arg.requires_grad:
                        if grad is not None:
                            arg.backward(grad.data if isinstance(grad, Value) else grad)
            result.grad_fn = backward_function
        return result

class Context:
    def save_for_backward(self, *args):
        self.saved_tensors = args

    def saved_tensors(self):
        return self.saved_tensors
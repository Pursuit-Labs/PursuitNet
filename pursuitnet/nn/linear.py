import numpy as np
from pursuitnet.tensor import Tensor

class Linear:
    def __init__(self, in_features, out_features, initial_weights=None, initial_bias=None):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(
            initial_weights if initial_weights is not None else np.random.randn(in_features, out_features).astype(np.float32) * np.sqrt(2. / in_features),
            requires_grad=True
        )
        self.bias = Tensor(
            initial_bias if initial_bias is not None else np.zeros(out_features, dtype=np.float32),
            requires_grad=True
        )

    def forward(self, input):
        output = input @ self.weight + self.bias
        
        if output.requires_grad:
            def _backward(grad_output):
                # Compute gradient for weights and bias
                if self.weight.requires_grad:
                    weight_grad = input.data.T @ grad_output
                    self.weight.backward(weight_grad)
                
                if self.bias.requires_grad:
                    bias_grad = np.sum(grad_output, axis=0)
                    self.bias.backward(bias_grad)
                
                # Compute gradient for input
                if input.requires_grad:
                    input_grad = grad_output @ self.weight.data.T
                    input.backward(input_grad)
            
            output._grad_fn = _backward

        return output

    def __call__(self, input):
        return self.forward(input)

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features})"
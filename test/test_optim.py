import unittest
import numpy as np
import platform
import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pursuitnet as pn

def tensor_sum(tensors):
    result = tensors[0]
    for tensor in tensors[1:]:
        result = result + tensor
    return result

class TestAdamOptimizer(unittest.TestCase):
    def setUp(self):
        self.params = [pn.Tensor(np.random.randn(10, 10), requires_grad=True) for _ in range(3)]
        self.adam_pn = pn.optim.Adam(self.params, lr=0.01)
        
        self.torch_params = [torch.tensor(param.data, requires_grad=True) for param in self.params]
        self.adam_torch = torch.optim.Adam(self.torch_params, lr=0.01)

    def assert_close(self, a, b, rtol=1e-5, atol=1e-8):
        a = a.data if isinstance(a, pn.Tensor) else a
        b = b.data if isinstance(b, torch.Tensor) else b
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        self.assertTrue(np.allclose(a, b, rtol=rtol, atol=atol))

    def test_initialization(self):
        self.assertEqual(self.adam_pn.lr, 0.01)
        self.assertEqual(self.adam_pn.betas, (0.9, 0.999))
        self.assertEqual(self.adam_pn.eps, 1e-8)

    def test_step(self):
        # Create a simple loss and backward pass
        loss_pn = tensor_sum([(param * param).sum() for param in self.params])
        loss_pn.backward()

        loss_torch = sum((param ** 2).sum() for param in self.torch_params)
        loss_torch.backward()

        # Step both optimizers
        self.adam_pn.step()
        self.adam_torch.step()

        # Compare parameters after step
        for pn_param, torch_param in zip(self.params, self.torch_params):
            self.assert_close(pn_param.data, torch_param.detach().numpy())

    def test_zero_grad(self):
        # Create gradients
        loss_pn = tensor_sum([(param * param).sum() for param in self.params])
        loss_pn.backward()

        # Zero out gradients
        self.adam_pn.zero_grad()

        # Check if gradients are zero
        for param in self.params:
            self.assertTrue(np.allclose(param.grad, 0))

    def test_multiple_steps(self):
        for _ in range(5):  # Perform 5 optimization steps
            # PursuitNet
            loss_pn = tensor_sum([(param * param).sum() for param in self.params])
            loss_pn.backward()
            self.adam_pn.step()
            self.adam_pn.zero_grad()

            # PyTorch
            loss_torch = sum((param ** 2).sum() for param in self.torch_params)
            loss_torch.backward()
            self.adam_torch.step()
            self.adam_torch.zero_grad()

        # Compare final parameters
        for pn_param, torch_param in zip(self.params, self.torch_params):
            self.assert_close(pn_param.data, torch_param.detach().numpy())

if __name__ == '__main__':
    print(f"Running tests on {platform.system()} {platform.machine()}")
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"PursuitNet version: {pn.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    unittest.main()
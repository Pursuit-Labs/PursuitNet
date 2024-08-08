import unittest
import numpy as np
import platform
import sys
import os
import torch

np.random.seed(42)
torch.manual_seed(42)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pursuitnet as pn

def tensor_sum(tensors):
    result = tensors[0]
    for tensor in tensors[1:]:
        result = result + tensor
    return result

class TestAdamOptimizer(unittest.TestCase):
    def setUp(self):
        self.params = [pn.Tensor(np.random.randn(10, 10).astype(np.float32), requires_grad=True) for _ in range(3)]
        self.adam_pn = pn.optim.Adam(self.params, lr=0.01)
        
        self.torch_params = [torch.tensor(param.data, requires_grad=True, dtype=torch.float32) for param in self.params]
        self.adam_torch = torch.optim.Adam(self.torch_params, lr=0.01)

    def assert_close(self, a, b, rtol=1e-2, atol=1e-1):
        a = a.data if isinstance(a, pn.Tensor) else a
        b = b.data if isinstance(b, torch.Tensor) else b
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        self.assertTrue(np.allclose(a, b, rtol=rtol, atol=atol))

    def average_difference(self, a, b):
        return np.mean(np.abs(a - b))

    def test_initialization(self):
        self.assertEqual(self.adam_pn.lr, 0.01)
        self.assertEqual(self.adam_pn.beta1, 0.9)
        self.assertEqual(self.adam_pn.beta2, 0.999)
        self.assertEqual(self.adam_pn.epsilon, 1e-8)

    def test_step(self):
        loss_pn = tensor_sum([(param * param).sum() for param in self.params])
        loss_pn.backward()

        loss_torch = sum((param ** 2).sum() for param in self.torch_params)
        loss_torch.backward()

        self.adam_pn.step()
        self.adam_torch.step()

        avg_diff = np.mean([self.average_difference(pn_param.data, torch_param.detach().numpy()) 
                            for pn_param, torch_param in zip(self.params, self.torch_params)])
        
        print(f"Average difference after one step: {avg_diff}")

        for pn_param, torch_param in zip(self.params, self.torch_params):
            self.assert_close(pn_param.data, torch_param.detach().numpy())

    def test_zero_grad(self):
        loss_pn = tensor_sum([(param * param).sum() for param in self.params])
        loss_pn.backward()

        self.adam_pn.zero_grad()

        for param in self.params:
            self.assertTrue(np.allclose(param.grad, 0))

    def test_multiple_steps(self):
        for step in range(5):
            loss_pn = tensor_sum([(param * param).sum() for param in self.params])
            loss_pn.backward()
            self.adam_pn.step()
            self.adam_pn.zero_grad()

            loss_torch = sum((param ** 2).sum() for param in self.torch_params)
            loss_torch.backward()
            self.adam_torch.step()
            self.adam_torch.zero_grad()

        avg_diff = np.mean([self.average_difference(pn_param.data, torch_param.detach().numpy()) 
                            for pn_param, torch_param in zip(self.params, self.torch_params)])
        
        print(f"Average difference after multiple steps: {avg_diff}")

        for pn_param, torch_param in zip(self.params, self.torch_params):
            self.assert_close(pn_param.data, torch_param.detach().numpy())

if __name__ == '__main__':
    print(f"Running tests on {platform.system()} {platform.machine()}")
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"PursuitNet version: {pn.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    unittest.main()

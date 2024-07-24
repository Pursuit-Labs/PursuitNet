import unittest
import numpy as np
import platform
import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pursuitnet as pn

class TestCrossEntropyLoss(unittest.TestCase):
    def setUp(self):
        self.loss_fn_pn = pn.nn.CrossEntropyLoss()
        self.loss_fn_torch = torch.nn.CrossEntropyLoss()

    def assert_close(self, a, b, rtol=1e-5, atol=1e-8):
        a = a.data if isinstance(a, pn.Tensor) else a
        b = b.data if isinstance(b, pn.Tensor) else b
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        self.assertTrue(np.allclose(a, b, rtol=rtol, atol=atol))

    def test_forward(self):
        input_data = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        target_data = np.array([1, 2, 0])

        input_tensor_pn = pn.Tensor(input_data, requires_grad=True)
        target_tensor_pn = pn.Tensor(target_data)

        input_tensor_torch = torch.tensor(input_data, requires_grad=True)
        target_tensor_torch = torch.tensor(target_data)

        loss_pn = self.loss_fn_pn(input_tensor_pn, target_tensor_pn)
        loss_torch = self.loss_fn_torch(input_tensor_torch, target_tensor_torch)

        # Convert loss_pn.data to a scalar if it's not already
        loss_pn_value = loss_pn.data.item() if hasattr(loss_pn.data, 'item') else loss_pn.data

        self.assertAlmostEqual(loss_pn_value, loss_torch.item(), places=6)

    def test_backward(self):
        input_data = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]], dtype=np.float32)
        target_data = np.array([0, 1, 2], dtype=np.int64)
        
        input_tensor_pn = pn.Tensor(input_data, requires_grad=True)
        target_tensor_pn = pn.Tensor(target_data)
        
        input_tensor_torch = torch.tensor(input_data, requires_grad=True)
        target_tensor_torch = torch.tensor(target_data, dtype=torch.long)
        
        loss_pn = self.loss_fn_pn(input_tensor_pn, target_tensor_pn)
        loss_pn.backward()
        
        loss_torch = self.loss_fn_torch(input_tensor_torch, target_tensor_torch)
        loss_torch.backward()

        # Logging for debugging
        print("PyTorch Gradient:\n", input_tensor_torch.grad.numpy())
        print("PursuitNet Gradient:\n", input_tensor_pn.grad)
        
        self.assert_close(input_tensor_pn.grad, input_tensor_torch.grad.numpy())

if __name__ == '__main__':
    print(f"Running tests on {platform.system()} {platform.machine()}")
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    unittest.main()

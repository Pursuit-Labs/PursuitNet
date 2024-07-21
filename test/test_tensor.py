import unittest
import numpy as np
import platform
import sys
import torch
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pursuitnet as pn

def gpu_available():
    try:
        import cupy
        cupy.cuda.runtime.getDeviceCount()
        return True
    except:
        return False

class TestCPUTensor(unittest.TestCase):
    def setUp(self):
        self.data = [[1, 2], [3, 4]]
        self.cpu_tensor = pn.Tensor(self.data, dtype=pn.float32)

    def test_creation(self):
        self.assertIsInstance(self.cpu_tensor, pn.Tensor)

    def test_dtype(self):
        self.assertEqual(self.cpu_tensor.dtype, pn.float32)

    def test_device(self):
        self.assertEqual(self.cpu_tensor.device, 'cpu')

    def test_shape(self):
        self.assertEqual(str(self.cpu_tensor.shape), 'pursuitnet.Size([2, 2])')

    def test_indexing(self):
        self.assertEqual(self.cpu_tensor[0, 1].data, 2)

    def test_addition(self):
        result = self.cpu_tensor + 1
        self.assertTrue(np.array_equal(result.data, np.array([[2, 3], [4, 5]])))

    def test_subtraction(self):
        result = self.cpu_tensor - 1
        self.assertTrue(np.array_equal(result.data, np.array([[0, 1], [2, 3]])))

    def test_multiplication(self):
        result = self.cpu_tensor * 2
        self.assertTrue(np.array_equal(result.data, np.array([[2, 4], [6, 8]])))

    def test_division(self):
        result = self.cpu_tensor / 2
        self.assertTrue(np.array_equal(result.data, np.array([[0.5, 1], [1.5, 2]])))

    def test_mean(self):
        self.assertAlmostEqual(self.cpu_tensor.mean().data, 2.5)

    def test_sum(self):
        self.assertEqual(self.cpu_tensor.sum().data, 10)

    def test_max(self):
        self.assertEqual(self.cpu_tensor.max().data, 4)

    def test_min(self):
        self.assertEqual(self.cpu_tensor.min().data, 1)

    def test_reshape(self):
        reshaped = self.cpu_tensor.reshape((1, 4))
        self.assertEqual(str(reshaped.shape), 'pursuitnet.Size([1, 4])')

    def test_transpose(self):
        transposed = self.cpu_tensor.transpose()
        self.assertEqual(str(transposed.shape), 'pursuitnet.Size([2, 2])')
        self.assertTrue(np.array_equal(transposed.data, np.array([[1, 3], [2, 4]])))

    def test_matmul(self):
        result = self.cpu_tensor @ self.cpu_tensor
        self.assertTrue(np.array_equal(result.data, np.array([[7, 10], [15, 22]])))

    def test_broadcasting(self):
        broadcasted = self.cpu_tensor + pn.Tensor([1, 2])
        self.assertTrue(np.array_equal(broadcasted.data, np.array([[2, 4], [4, 6]])))

    def test_repr(self):
        self.assertIn("Tensor", repr(self.cpu_tensor))

    def test_str(self):
        self.assertIn("1", str(self.cpu_tensor))
        self.assertIn("4", str(self.cpu_tensor))

@unittest.skipIf(not gpu_available(), "GPU not available")
class TestGPUTensor(unittest.TestCase):
    def setUp(self):
        self.data = [[1, 2], [3, 4]]
        self.gpu_tensor = pn.Tensor(self.data, dtype=np.float32, device='gpu')

    def test_creation(self):
        self.assertIsInstance(self.gpu_tensor, pn.Tensor)

    def test_dtype(self):
        self.assertEqual(self.gpu_tensor.dtype, pn.float32)

    def test_device(self):
        self.assertEqual(self.gpu_tensor.device, 'gpu')

    def test_shape(self):
        self.assertEqual(self.gpu_tensor.shape, (2, 2))

    def test_to_device(self):
        cpu_tensor = self.gpu_tensor.to('cpu')
        self.assertEqual(cpu_tensor.device, 'cpu')

    def test_indexing(self):
        self.assertEqual(self.gpu_tensor[1, 0].data, 3)

    # Add more GPU-specific tests here...

class TestTensorRepr(unittest.TestCase):
    def assert_repr_match(self, pn_tensor, pt_tensor):
        pn_repr = repr(pn_tensor)
        pt_repr = repr(pt_tensor)
        
        # Replace 'tensor' with 'Tensor' and 'torch' with 'pursuitnet'
        pt_repr = pt_repr.replace('tensor', 'Tensor').replace('torch', 'pursuitnet')
        
        self.assertEqual(pn_repr, pt_repr)

    def test_float32_no_decimal(self):
        data = [[1, 2], [3, 4]]
        pn_tensor = pn.Tensor(data, dtype=pn.float32)
        pt_tensor = torch.tensor(data, dtype=torch.float32)
        self.assert_repr_match(pn_tensor, pt_tensor)

    def test_float32_with_decimal(self):
        data = [[1.5, 2.7], [3.1, 4.9]]
        pn_tensor = pn.Tensor(data, dtype=pn.float32)
        pt_tensor = torch.tensor(data, dtype=torch.float32)
        self.assert_repr_match(pn_tensor, pt_tensor)

    def test_float32_many_decimal_places(self):
        data = [[1.123456, 2.987654], [3.141592, 4.654321]]
        pn_tensor = pn.Tensor(data, dtype=pn.float32)
        pt_tensor = torch.tensor(data, dtype=torch.float32)
        self.assert_repr_match(pn_tensor, pt_tensor)

    def test_int32(self):
        data = [[1, 2], [3, 4]]
        pn_tensor = pn.Tensor(data, dtype=pn.int32)
        pt_tensor = torch.tensor(data, dtype=torch.int32)
        self.assert_repr_match(pn_tensor, pt_tensor)

    def test_bool(self):
        data = [[True, False], [False, True]]
        pn_tensor = pn.Tensor(data, dtype=pn.bool)
        pt_tensor = torch.tensor(data, dtype=torch.bool)
        self.assert_repr_match(pn_tensor, pt_tensor)

    def test_float64(self):
        data = [[1.23456789, 2.98765432], [3.14159265, 4.65432198]]
        pn_tensor = pn.Tensor(data, dtype=pn.float64)
        pt_tensor = torch.tensor(data, dtype=torch.float64)
        self.assert_repr_match(pn_tensor, pt_tensor)

    def test_large_tensor(self):
        data = np.random.rand(10, 10)
        pn_tensor = pn.Tensor(data, dtype=pn.float32)
        pt_tensor = torch.tensor(data, dtype=torch.float32)
        self.assert_repr_match(pn_tensor, pt_tensor)

    def test_small_values(self):
        data = [[1e-8, 2e-8], [3e-8, 4e-8]]
        pn_tensor = pn.Tensor(data, dtype=pn.float32)
        pt_tensor = torch.tensor(data, dtype=torch.float32)
        self.assert_repr_match(pn_tensor, pt_tensor)

    def test_large_values(self):
        data = [[1e8, 2e8], [3e8, 4e8]]
        pn_tensor = pn.Tensor(data, dtype=pn.float32)
        pt_tensor = torch.tensor(data, dtype=torch.float32)
        self.assert_repr_match(pn_tensor, pt_tensor)

class TestTensorSize(unittest.TestCase):
    def test_tensor_shape(self):
        data = [[1, 2], [3, 4]]
        tensor = pn.Tensor(data, dtype=pn.float32)
        self.assertEqual(str(tensor.shape), 'pursuitnet.Size([2, 2])')

    def test_reshape_tensor(self):
        data = [[1, 2], [3, 4]]
        tensor = pn.Tensor(data, dtype=pn.float32)
        reshaped = tensor.reshape(1, 4)
        self.assertEqual(str(reshaped.shape), 'pursuitnet.Size([1, 4])')

    def test_transpose_tensor(self):
        data = [[1, 2], [3, 4]]
        tensor = pn.Tensor(data, dtype=pn.float32)
        transposed = tensor.transpose()
        self.assertEqual(str(transposed.shape), 'pursuitnet.Size([2, 2])')

class TestGradients(unittest.TestCase):
    def assert_close(self, a, b, rtol=1e-5, atol=1e-8):
        def to_list(x):
            if x is None:
                return None
            if isinstance(x, pn.Tensor):
                return x.data.tolist() if x.data is not None else None
            elif isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy().tolist()
            elif isinstance(x, np.ndarray):
                return x.tolist()
            else:
                return list(x)

        a_list = to_list(a)
        b_list = to_list(b)

        if a_list is None and b_list is None:
            return
        
        self.assertIsNotNone(a_list, "First argument is None")
        self.assertIsNotNone(b_list, "Second argument is None")

        self.assertEqual(len(a_list), len(b_list), "Arrays have different lengths")

        for a_val, b_val in zip(a_list, b_list):
            if isinstance(a_val, (list, tuple)) and isinstance(b_val, (list, tuple)):
                self.assert_close(a_val, b_val, rtol, atol)
            else:
                self.assertAlmostEqual(a_val, b_val, delta=max(rtol * abs(b_val), atol),
                                    msg=f"Values not close: {a_val} != {b_val}")
                
    def test_requires_grad(self):
        pn_tensor = pn.Tensor([1.0, 2.0, 3.0], requires_grad=True)
        pt_tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        self.assertTrue(pn_tensor.requires_grad)
        self.assertTrue(pt_tensor.requires_grad)

    def test_grad_none_initially(self):
        pn_tensor = pn.Tensor([1.0, 2.0, 3.0], requires_grad=True)
        pt_tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        self.assertIsNone(pn_tensor.grad)
        self.assertIsNone(pt_tensor.grad)

    def test_simple_backward(self):
        pn_tensor = pn.Tensor([1.0, 2.0, 3.0], requires_grad=True)
        pt_tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        pn_result = pn_tensor.sum()
        pt_result = pt_tensor.sum()
        
        pn_result.backward()
        pt_result.backward()
        
        self.assert_close(pn_tensor.grad, pt_tensor.grad)

    def test_addition_backward(self):
        pn_x = pn.Tensor([1.0, 2.0, 3.0], requires_grad=True)
        pn_y = pn.Tensor([4.0, 5.0, 6.0], requires_grad=True)
        pt_x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        pt_y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
        
        pn_z = (pn_x + pn_y).sum()
        pt_z = (pt_x + pt_y).sum()
        
        pn_z.backward()
        pt_z.backward()
        
        self.assert_close(pn_x.grad, pt_x.grad)
        self.assert_close(pn_y.grad, pt_y.grad)

    def test_multiplication_backward(self):
        pn_x = pn.Tensor([1.0, 2.0, 3.0], requires_grad=True)
        pn_y = pn.Tensor([4.0, 5.0, 6.0], requires_grad=True)
        pt_x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        pt_y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
        
        pn_z = (pn_x * pn_y).sum()
        pt_z = (pt_x * pt_y).sum()
        
        pn_z.backward()
        pt_z.backward()
        
        self.assert_close(pn_x.grad, pt_x.grad)
        self.assert_close(pn_y.grad, pt_y.grad)

    def test_matmul_backward(self):
        pn_x = pn.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        pn_y = pn.Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        pt_x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        pt_y = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        
        pn_z = (pn_x @ pn_y).sum()
        pt_z = (pt_x @ pt_y).sum()
        
        pn_z.backward()
        pt_z.backward()
        
        self.assert_close(pn_x.grad, pt_x.grad)
        self.assert_close(pn_y.grad, pt_y.grad)

    def test_zero_grad(self):
        pn_tensor = pn.Tensor([1.0, 2.0, 3.0], requires_grad=True)
        pt_tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        pn_result = pn_tensor.sum()
        pt_result = pt_tensor.sum()
        
        pn_result.backward()
        pt_result.backward()
        
        self.assertIsNotNone(pn_tensor.grad, "pn_tensor.grad should not be None after backward")
        self.assertIsNotNone(pt_tensor.grad, "pt_tensor.grad should not be None after backward")
        
        pn_tensor.zero_grad()
        pt_tensor.grad.zero_()
        
        self.assert_close(pn_tensor.grad, pt_tensor.grad)

    def test_complex_computation(self):
        pn_x = pn.Tensor([1.0, 2.0, 3.0], requires_grad=True)
        pn_y = pn.Tensor([4.0, 5.0, 6.0], requires_grad=True)
        pt_x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        pt_y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
        
        pn_z = ((pn_x * pn_y).sum() + (pn_x + pn_y).sum()) * pn_x.sum()
        pt_z = ((pt_x * pt_y).sum() + (pt_x + pt_y).sum()) * pt_x.sum()
        
        pn_z.backward()
        pt_z.backward()
        
        self.assert_close(pn_x.grad, pt_x.grad)
        self.assert_close(pn_y.grad, pt_y.grad)

if __name__ == '__main__':
    print(f"Running tests on {platform.system()} {platform.machine()}")
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"GPU available: {gpu_available()}")
    unittest.main()
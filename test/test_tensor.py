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
        self.assertEqual(self.cpu_tensor.shape, (2, 2))

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
        self.assertEqual(reshaped.shape, (1, 4))

    def test_transpose(self):
        transposed = self.cpu_tensor.transpose()
        self.assertEqual(transposed.shape, (2, 2))
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

if __name__ == '__main__':
    print(f"Running tests on {platform.system()} {platform.machine()}")
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"GPU available: {gpu_available()}")
    unittest.main()
import unittest
import numpy as np
import torch
import torch.nn as torch_nn
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pursuitnet.nn as pn_nn
import pursuitnet as pn

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class TestModelCreation(unittest.TestCase):
    def setUp(self):
        # Initialize common weights and biases
        self.custom_fc1_weights = np.random.normal(0.0, np.sqrt(2.0 / (28 * 28)), (512, 28 * 28)).astype(np.float32)
        self.custom_fc1_bias = np.zeros(512, dtype=np.float32)
        self.custom_fc2_weights = np.random.normal(0.0, np.sqrt(2.0 / 512), (10, 512)).astype(np.float32)
        self.custom_fc2_bias = np.zeros(10, dtype=np.float32)

        # PyTorch model definition
        class PyTorchNet(torch_nn.Module):
            def __init__(self, fc1_weights, fc1_bias, fc2_weights, fc2_bias):
                super(PyTorchNet, self).__init__()
                self.fc1 = torch_nn.Linear(28 * 28, 512)
                self.fc2 = torch_nn.Linear(512, 10)
                self.init_weights(fc1_weights, fc1_bias, fc2_weights, fc2_bias)

            def init_weights(self, fc1_weights, fc1_bias, fc2_weights, fc2_bias):
                with torch.no_grad():
                    self.fc1.weight = torch.nn.Parameter(torch.tensor(fc1_weights, dtype=torch.float32))
                    self.fc1.bias = torch.nn.Parameter(torch.tensor(fc1_bias, dtype=torch.float32))
                    self.fc2.weight = torch.nn.Parameter(torch.tensor(fc2_weights, dtype=torch.float32))
                    self.fc2.bias = torch.nn.Parameter(torch.tensor(fc2_bias, dtype=torch.float32))

            def forward(self, x):
                x = x.view(-1, 28 * 28)
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        # Custom framework model definition
        class CustomNet(pn_nn.Module):
            def __init__(self, fc1_weights, fc1_bias, fc2_weights, fc2_bias):
                super(CustomNet, self).__init__()
                self.fc1 = pn_nn.Linear(28 * 28, 512)
                self.fc2 = pn_nn.Linear(512, 10)
                self.relu = pn_nn.ReLU()  # Use ReLU from activation.py
                self.init_weights(fc1_weights, fc1_bias, fc2_weights, fc2_bias)

            def init_weights(self, fc1_weights, fc1_bias, fc2_weights, fc2_bias):
                # Correctly set parameters as Tensor objects
                self.fc1.weight.data = fc1_weights
                self.fc1.bias.data = fc1_bias
                self.fc2.weight.data = fc2_weights
                self.fc2.bias.data = fc2_bias

            def forward(self, x):
                if not isinstance(x, pn.Tensor):
                    x = pn.Tensor(x)  # Ensure x is a PursuitNet Tensor

                x = x.reshape(-1, 28 * 28)  # Using reshape method from Tensor class
                x = self.relu(self.fc1(x))  # Use the ReLU activation class
                x = self.fc2(x)
                return x

        self.pytorch_model = PyTorchNet(
            self.custom_fc1_weights,
            self.custom_fc1_bias,
            self.custom_fc2_weights,
            self.custom_fc2_bias,
        )
        self.custom_model = CustomNet(
            self.custom_fc1_weights,
            self.custom_fc1_bias,
            self.custom_fc2_weights,
            self.custom_fc2_bias,
        )

        # Test input
        self.input_data = np.random.rand(1, 28, 28).astype(np.float32)

        # PyTorch input
        self.pytorch_input = torch.tensor(self.input_data).float()

        # Custom framework input
        self.custom_input = pn.Tensor(self.input_data)

    def test_forward_output(self):
        # Get PyTorch output
        self.pytorch_model.eval()  # Set PyTorch model to evaluation mode
        with torch.no_grad():
            pytorch_output = self.pytorch_model(self.pytorch_input).numpy()

        # Get custom framework output
        custom_output = self.custom_model(self.custom_input)

        # Compare outputs with adjusted tolerances
        np.testing.assert_allclose(custom_output.data, pytorch_output, rtol=1e-4, atol=1e-6)

    def test_intermediate_activation(self):
        # Define hooks to capture intermediate activations
        self.pytorch_intermediate_activations = []

        def pytorch_hook(module, input, output):
            self.pytorch_intermediate_activations.append(output.detach().numpy())

        self.pytorch_model.fc1.register_forward_hook(pytorch_hook)
        self.pytorch_model.fc2.register_forward_hook(pytorch_hook)

        # Get PyTorch output to trigger hooks
        self.pytorch_model.eval()  # Set PyTorch model to evaluation mode
        with torch.no_grad():
            self.pytorch_model(self.pytorch_input)

        # Custom framework intermediate activations
        custom_intermediate_activations = []

        x = self.custom_input
        x = x.reshape(-1, 28 * 28)
        custom_intermediate_activations.append(self.custom_model.fc1(x))
        x = self.custom_model.relu(custom_intermediate_activations[-1])  # Use the ReLU activation class
        custom_intermediate_activations.append(self.custom_model.fc2(x))

        # Compare intermediate activations with adjusted tolerances
        for idx, custom_activation in enumerate(custom_intermediate_activations):
            pytorch_activation = self.pytorch_intermediate_activations[idx]
            np.testing.assert_allclose(custom_activation.data, pytorch_activation, rtol=1e-4, atol=1e-6)

if __name__ == '__main__':
    unittest.main()

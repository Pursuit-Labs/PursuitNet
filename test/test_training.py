import unittest
import numpy as np
import sys
import os

# Add project path to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pursuitnet as pn
from pursuitnet.nn import Linear, ReLU, Sequential
from pursuitnet.optim import Adam

class TestTrainingLoop(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 10).astype(np.float32)
        self.y = np.random.randint(0, 2, size=(100,))
        self.model = pn.nn.Sequential(
            Linear(10, 50),
            ReLU(),
            Linear(50, 2)
        )
        self.loss_fn = pn.nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.01)

    def test_training_loop(self):
        print("\nSetting up test...")
        print(f"Data shape: X: {self.X.shape}, y: {self.y.shape}")
        print(f"Model architecture: {self.model}")
        print(f"Total parameters: {len(list(self.model.parameters()))}")
        print(f"Optimizer: {self.optimizer}, Learning rate: {self.optimizer.lr}")

        print("Starting training loop...")
        batch_size = 10
        num_epochs = 10

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            for i in range(0, len(self.X), batch_size):
                X_batch = pn.Tensor(self.X[i:i+batch_size], requires_grad=True)
                y_batch = pn.Tensor(self.y[i:i+batch_size])

                print(f"  Batch {i // batch_size + 1}: X shape: {X_batch.shape}, y shape: {y_batch.shape}")

                # Forward pass
                predictions = self.model(X_batch)
                for j, layer in enumerate(self.model.layers):
                    print(f"Layer {j} output shape: {predictions.shape}")

                print(f"  Predictions shape: {predictions.shape}")

                # Compute loss
                loss = self.loss_fn(predictions, y_batch)
                print(f"  Batch loss: {loss.data.item()}")

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Update parameters
                self.optimizer.step()

            # Break after the first epoch for testing purposes
            break

        # Add assertions here to check if the training loop is working correctly
        self.assertIsNotNone(loss.data)
        self.assertGreater(loss.data.item(), 0)

    def test_zero_grad(self):
        print("\nSetting up test...")
        print(f"Data shape: X: {self.X.shape}, y: {self.y.shape}")
        print(f"Model architecture: {self.model}")
        print(f"Total parameters: {len(list(self.model.parameters()))}")
        print(f"Optimizer: {self.optimizer}, Learning rate: {self.optimizer.lr}")

        print("Testing zero_grad...")
        # Set some non-zero gradients
        for param in self.model.parameters():
            param.grad = np.ones_like(param.data)

        # Call zero_grad
        self.optimizer.zero_grad()

        # Check if all gradients are zero
        print(f"Total parameters: {len(list(self.model.parameters()))}")
        for i, param in enumerate(self.model.parameters()):
            print(f"Parameter {i} grad is {param.grad}")
            self.assertIsNotNone(param.grad)
            self.assertTrue(np.allclose(param.grad, 0))

if __name__ == '__main__':
    unittest.main()
import unittest
import numpy as np
import sys
import os

# Add project path to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pursuitnet as pn
import pursuitnet.nn as nn

class TestTrainingLoop(unittest.TestCase):
    def setUp(self):
        print("\nSetting up test...")
        # Example data
        self.X = np.random.randn(100, 10)
        self.y = np.random.randint(0, 2, size=(100,))  # Assume binary classification
        print(f"Data shape: X: {self.X.shape}, y: {self.y.shape}")

        # Model setup
        self.model = [
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 2)  # Binary classification with 2 classes
        ]
        print("Model architecture:", self.model)

        # Loss function and optimizer
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = pn.optim.Adam(self.model_parameters(), lr=0.01)
        print(f"Optimizer: {self.optimizer}, Learning rate: {self.optimizer.lr}")

    def model_parameters(self):
        params = []
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                params.extend([layer.weight, layer.bias])
        print(f"Total parameters: {len(params)}")
        return params

    def forward_pass(self, Xbatch):
        output = pn.Tensor(Xbatch)
        for i, layer in enumerate(self.model):
            output = layer(output)
            print(f"Layer {i} output shape: {output.shape}")
        return output

    def test_training_loop(self):
        print("\nStarting training loop...")
        n_epochs = 10
        batch_size = 10
        last_epoch_loss = None
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch + 1}/{n_epochs}")
            epoch_losses = []
            for i in range(0, len(self.X), batch_size):
                Xbatch = pn.Tensor(self.X[i:i + batch_size], requires_grad=True)
                ybatch = pn.Tensor(self.y[i:i + batch_size].flatten(), requires_grad=False)
                print(f"  Batch {i//batch_size + 1}: X shape: {Xbatch.shape}, y shape: {ybatch.shape}")

                # Forward pass
                y_pred = self.forward_pass(Xbatch)
                print(f"  Predictions shape: {y_pred.shape}")

                # Compute loss
                loss = self.loss_fn(y_pred, ybatch)
                print(f"  Batch loss: {loss.data.item()}")
                epoch_losses.append(loss.data.item())

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                
                # Print gradients
                print("  Gradients:")
                for j, param in enumerate(self.model_parameters()):
                    if param.grad is not None:
                        print(f"    Param {j} grad norm: {np.linalg.norm(param.grad)}")
                    else:
                        print(f"    Param {j} grad is None")
                
                self.optimizer.step()

            avg_epoch_loss = np.mean(epoch_losses)
            print(f'Finished epoch {epoch + 1}, average loss {avg_epoch_loss}')

            if last_epoch_loss is not None:
                print(f"  Previous epoch loss: {last_epoch_loss}")
                print(f"  Current epoch loss: {avg_epoch_loss}")
                print(f"  Loss change: {last_epoch_loss - avg_epoch_loss}")
                self.assertLess(avg_epoch_loss, last_epoch_loss, "Loss did not decrease; training may not be effective.")
                self.assertNotEqual(last_epoch_loss, avg_epoch_loss, "Loss did not change; weights may not be updating.")

            last_epoch_loss = avg_epoch_loss

    def test_zero_grad(self):
        print("\nTesting zero_grad...")
        self.optimizer.zero_grad()
        for i, param in enumerate(self.model_parameters()):
            if param.grad is None:
                print(f"Parameter {i} grad is None")
            else:
                print(f"Parameter {i} grad norm: {np.linalg.norm(param.grad)}")
            self.assertTrue(param.grad is None or np.all(param.grad == 0), "Gradients should be zero after zero_grad call.")

if __name__ == '__main__':
    unittest.main(verbosity=2)
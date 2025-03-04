import torch.nn as nn


class LogisticRegression(nn.Module):
    """
    A simple implementation of a Logistic Regression model using PyTorch for MNIST dataset.

    This model performs linear classification on input data by applying a linear transformation.

    Attributes:
    - linear (nn.Linear): A linear transformation layer with 784 input features and 10 output features.
    """

    def __init__(self):
        """
        Initializes the LogisticRegression model.

        The model consists of a single linear layer that maps an input of size 784
        (28x28 flattened image, for example) to 10 output classes.
        """
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
        - x (torch.Tensor): The input tensor of shape `(batch_size, 1, 28, 28)` or
          `(batch_size, 28, 28)` for MNIST dataset. This tensor is reshaped to `(batch_size, 784)`.

        Returns:
        - torch.Tensor: The output tensor of shape `(batch_size, 10)` containing raw class scores.
        """
        # Flatten the input tensor to a 2D tensor with size (batch_size, 784)
        x = x.view(x.size(0), -1)

        # Apply the linear transformation
        out = self.linear(x)

        return out

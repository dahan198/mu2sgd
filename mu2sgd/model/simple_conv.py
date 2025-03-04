import torch.nn as nn
import torch.nn.functional as F


class SimpleConv(nn.Module):
    """
    A simple convolutional neural network for MNIST classification tasks.

    This network consists of two convolutional layers, followed by max pooling,
    and two fully connected layers with batch normalization applied to the first
    fully connected layer.

    Attributes:
    - _c1 (nn.Conv2d): First convolutional layer with 20 filters and a 5x5 kernel.
    - _c2 (nn.Conv2d): Second convolutional layer with 50 filters and a 5x5 kernel.
    - _f1 (nn.Linear): Fully connected layer mapping 800 features to 50.
    - _bn (nn.BatchNorm1d): Batch normalization layer for 50 features.
    - _f2 (nn.Linear): Fully connected layer mapping 50 features to 10 (number of output classes).
    """

    def __init__(self):
        """
        Initializes the SimpleConv model.

        The model architecture includes:
        - Two 2D convolutional layers with ReLU activations and max pooling.
        - A fully connected layer with batch normalization and ReLU activation.
        - A final fully connected layer for classification.
        """
        super().__init__()
        self._c1 = nn.Conv2d(1, 20, kernel_size=5, stride=1)
        self._c2 = nn.Conv2d(20, 50, kernel_size=5, stride=1)
        self._f1 = nn.Linear(800, 50)
        self._bn = nn.BatchNorm1d(50)
        self._f2 = nn.Linear(50, 10)

    def forward(self, x):
        """
        Defines the forward pass of the SimpleConv model.

        Args:
        - x (torch.Tensor): Input tensor of shape `(batch_size, 1, H, W)` where
          H and W are the height and width of the input image (e.g., 28x28 for MNIST).

        Returns:
        - torch.Tensor: Output tensor of shape `(batch_size, 10)` containing raw class scores.
        """
        # Apply the first convolutional layer and ReLU activation
        x = F.relu(self._c1(x))

        # Apply max pooling with a 2x2 kernel and stride 2
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Apply the second convolutional layer and ReLU activation
        x = F.relu(self._c2(x))

        # Apply max pooling with a 2x2 kernel and stride 2
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Flatten the feature maps into a vector of size 800
        x = x.view(-1, 800)

        # Apply the first fully connected layer with batch normalization and ReLU activation
        x = F.relu(self._bn(self._f1(x)))

        # Apply the final fully connected layer for classification
        x = self._f2(x)

        return x

import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    A basic residual block used in ResNet architectures.

    Attributes:
    - expansion (int): The expansion factor for output channels. For BasicBlock, it's 1.
    - conv1 (nn.Conv2d): First 3x3 convolutional layer.
    - bn1 (nn.BatchNorm2d): Batch normalization layer for the first convolution.
    - conv2 (nn.Conv2d): Second 3x3 convolutional layer.
    - bn2 (nn.BatchNorm2d): Batch normalization layer for the second convolution.
    - shortcut (nn.Sequential): Shortcut connection for matching dimensions when necessary.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        """
        Initializes a BasicBlock.

        Args:
        - in_planes (int): Number of input channels.
        - planes (int): Number of output channels.
        - stride (int, optional): Stride for the convolution. Default is 1.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        """
        Defines the forward pass of the BasicBlock.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, in_planes, H, W).

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, planes, H/stride, W/stride).
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    A bottleneck residual block used in deeper ResNet architectures.

    Attributes:
    - expansion (int): The expansion factor for output channels. For Bottleneck, it's 4.
    - conv1 (nn.Conv2d): First 1x1 convolutional layer for reducing dimensions.
    - bn1 (nn.BatchNorm2d): Batch normalization layer for the first convolution.
    - conv2 (nn.Conv2d): Second 3x3 convolutional layer for spatial processing.
    - bn2 (nn.BatchNorm2d): Batch normalization layer for the second convolution.
    - conv3 (nn.Conv2d): Third 1x1 convolutional layer for restoring dimensions.
    - bn3 (nn.BatchNorm2d): Batch normalization layer for the third convolution.
    - shortcut (nn.Sequential): Shortcut connection for matching dimensions when necessary.
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        """
        Initializes a Bottleneck block.

        Args:
        - in_planes (int): Number of input channels.
        - planes (int): Number of intermediate channels.
        - stride (int, optional): Stride for the convolution. Default is 1.
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        """
        Defines the forward pass of the Bottleneck block.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, in_planes, H, W).

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, planes*expansion, H/stride, W/stride).
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    A ResNet architecture built with BasicBlock or Bottleneck blocks.

    Attributes:
    - conv1 (nn.Conv2d): Initial convolutional layer.
    - bn1 (nn.BatchNorm2d): Batch normalization layer for the initial convolution.
    - layer1, layer2, layer3, layer4 (nn.Sequential): Stacked residual layers.
    - linear (nn.Linear): Fully connected layer for classification.
    """
    def __init__(self, block, num_blocks, num_classes=10):
        """
        Initializes a ResNet model.

        Args:
        - block (class): Type of residual block (BasicBlock or Bottleneck).
        - num_blocks (list): Number of blocks in each layer.
        - num_classes (int, optional): Number of output classes. Default is 10.
        """
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        Creates a layer with stacked residual blocks.

        Args:
        - block (class): Type of residual block (BasicBlock or Bottleneck).
        - planes (int): Number of output channels for the layer.
        - num_blocks (int): Number of blocks to stack in the layer.
        - stride (int): Stride for the first block in the layer.

        Returns:
        - nn.Sequential: A sequential container of residual blocks.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Defines the forward pass of the ResNet model.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# ResNet model definitions with different depths
def ResNet9():
    """Creates a ResNet-9 model using BasicBlock."""
    return ResNet(BasicBlock, [1, 1, 1, 1])


def ResNet18():
    """Creates a ResNet-18 model using BasicBlock."""
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    """Creates a ResNet-34 model using BasicBlock."""
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    """Creates a ResNet-50 model using Bottleneck."""
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    """Creates a ResNet-101 model using Bottleneck."""
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    """Creates a ResNet-152 model using Bottleneck."""
    return ResNet(Bottleneck, [3, 8, 36, 3])

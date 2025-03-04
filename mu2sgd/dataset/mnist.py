import torchvision
import torchvision.transforms as transforms


class MNIST:
    """
    A wrapper class for loading and preprocessing the MNIST dataset.

    This class provides training and testing datasets with appropriate transformations
    applied to prepare the data for use in PyTorch models. It includes normalization
    for standardizing the input data.

    Attributes:
    - trainset (torchvision.datasets.MNIST): The training dataset with transformations applied.
    - testset (torchvision.datasets.MNIST): The testing dataset with transformations applied.
    """

    def __init__(self):
        """
        Initializes the MNIST class.

        The class sets up training and testing datasets with a common set of transformations.
        The transformations include converting images to tensors and normalizing them using
        the mean and standard deviation of the MNIST dataset.
        """
        super().__init__()

        # Define transformations for the dataset
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL images to tensors
            transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST mean and std
        ])

        # Load the training dataset with transformations
        self.trainset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )

        # Load the testing dataset with transformations
        self.testset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )

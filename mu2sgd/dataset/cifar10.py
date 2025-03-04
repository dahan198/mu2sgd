import torchvision
import torchvision.transforms as transforms


class CIFAR10:
    """
    A wrapper class for loading and preprocessing the CIFAR-10 dataset.

    This class defines training and testing datasets with specific data augmentation
    and normalization transformations applied to the images. It provides the datasets
    in a format ready for use with PyTorch DataLoaders.

    Attributes:
    - trainset (torchvision.datasets.CIFAR10): The training dataset with data augmentations.
    - testset (torchvision.datasets.CIFAR10): The testing dataset with normalization applied.
    """

    def __init__(self):
        """
        Initializes the CIFAR10 class.

        The class sets up training and testing datasets with appropriate transformations.
        Data augmentation is applied to the training dataset, while only normalization
        is applied to the testing dataset.
        """
        super().__init__()

        # Define transformations for the training set
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=2),  # Randomly crop to 32x32 with 2-pixel padding
            transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
            transforms.ToTensor(),  # Convert PIL images to tensors
            transforms.Normalize((0.4914, 0.4822, 0.4465),  # Normalize with CIFAR-10 mean and std
                                 (0.2023, 0.1994, 0.2010)),
        ])

        # Define transformations for the testing set
        transform_test = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL images to tensors
            transforms.Normalize((0.4914, 0.4822, 0.4465),  # Normalize with CIFAR-10 mean and std
                                 (0.2023, 0.1994, 0.2010)),
        ])

        # Load the training dataset with transformations
        self.trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )

        # Load the testing dataset with transformations
        self.testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )

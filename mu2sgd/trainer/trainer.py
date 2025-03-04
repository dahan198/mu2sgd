import os
import json
import csv
import wandb
import yaml
import torch
from ..utils import get_device, proj


class Trainer:
    """
    A class to manage the training and evaluation workflow for a PyTorch model.

    Attributes:
    - model (nn.Module): The PyTorch model to train.
    - optimizer (torch.optim.Optimizer): The optimizer for model training.
    - train_dataloader (DataLoader): DataLoader for training data.
    - test_dataloader (DataLoader): DataLoader for testing/validation data.
    - params (Namespace): A set of hyperparameters and configuration options.
    - device (torch.device): The device on which computations will be performed (CPU or GPU).
    - checkpoint_path (str): Directory to save model checkpoints.
    - log_file_path (str): Path to save training logs.
    - metrics_data (list): A list to store metrics data for later saving.
    - run_directory (str): Directory to store run-specific results.
    - use_wandb (bool): Whether to use Weights & Biases for experiment tracking.
    """

    def __init__(self, model, optimizer, train_dataloader, test_dataloader, params):
        """
        Initializes the Trainer class.

        Args:
        - model (nn.Module): The PyTorch model to train.
        - optimizer (torch.optim.Optimizer): Optimizer to use for training.
        - train_dataloader (DataLoader): DataLoader for training data.
        - test_dataloader (DataLoader): DataLoader for testing/validation data.
        - params (Namespace): Hyperparameters and configurations for training.
        """
        self.model = model
        self.criterion = None
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.device = get_device()
        self.model.to(self.device)
        self.checkpoint_path = None
        self.log_file_path = None
        self.params = params.__dict__
        self.metrics_data = []
        self.run_directory = "./"
        self.use_wandb = self.params["use_wandb"]

        if self.use_wandb:
            self.wandb = wandb
            with open(os.path.join(params.config_folder_path, "wandb.yaml"), 'r') as file:
                self.wandb_conf = yaml.safe_load(file)
            self.wandb.init(
                project=self.wandb_conf["project"],
                entity=self.wandb_conf["entity"],
                name=(
                    f"{self.params['optimizer']}--{self.params['dataset']}--{self.params['model']}-"
                    f"LR: {self.params['learning_rate']}--use-alpha_t: {self.params['use_alpha_t']}--"
                    f"Momentum: {self.params['gradient_momentum']}--Seed: {self.params['seed']}"
                ),
            )
            self.wandb.config.update(self.params)

    def train(self):
        """
        Prepares the environment for training, including creating directories for saving results,
        checkpoints, and parameters. This method sets up a unique run directory for each training run.
        """
        if not os.path.exists('results'):
            os.makedirs('results')

        # Find the next available run number
        existing_folders = [d for d in os.listdir('results') if os.path.isdir(os.path.join('results', d))]
        highest_num = len(existing_folders)
        next_folder_num = highest_num + 1
        run_directory = os.path.join('results', f'run{next_folder_num}')
        os.makedirs(run_directory)
        self.checkpoint_path = os.path.join(run_directory, "checkpoints")
        os.makedirs(self.checkpoint_path, exist_ok=True)
        self.log_file_path = os.path.join(run_directory, 'metrics_log.txt')
        self.run_directory = run_directory

        # Save hyperparameters to a JSON file
        params_file_path = os.path.join(run_directory, 'params.json')
        with open(params_file_path, 'w') as params_file:
            json.dump(self.params, params_file, indent=4)

    def evaluate(self):
        """
        Placeholder for evaluation logic. Computes validation metrics on the test dataset.
        """
        pass  # To be implemented based on specific evaluation metrics.

    def save_metrics_and_params(self):
        """
        Saves metrics and parameters to a CSV file in the run directory.

        If no metrics data is available, the method exits early.
        """
        csv_file_path = os.path.join(self.run_directory, 'results.csv')
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

        if not self.metrics_data:
            print("No metrics data to save.")
            return

        # Combine metrics and parameters into a single data structure
        combined_rows = [{**metrics, **self.params} for metrics in self.metrics_data]
        fieldnames = list(combined_rows[0].keys())

        # Write combined data to a CSV file
        with open(csv_file_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined_rows)

    def make_optimization_step(self, inputs, targets, first_step=False):
        """
        Performs a single optimization step, including forward pass, loss computation, and backpropagation.

        Handles specific cases for optimizers like `Mu2-SGD Experiments Toolkit` and `STORM`, with optional projection constraints.

        Args:
        - inputs (torch.Tensor): Input data batch.
        - targets (torch.Tensor): Ground truth labels.
        - first_step (bool, optional): Indicates whether it is the first optimization step. Default is False.

        Returns:
        - outputs (torch.Tensor): Model outputs from the forward pass.
        - loss (torch.Tensor): Computed loss for the current step.
        """
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        outputs = None
        loss = None

        if self.optimizer.__class__.__name__ in ['Mu2SGD', 'STORM']:
            if first_step:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.compute_estimator()
            else:
                # Standard optimization step with projection constraints
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                if self.params['projection_radius'] is not None:
                    with torch.no_grad():
                        for param in self.model.parameters():
                            param.data = proj(param.data, self.params['projection_radius'])

                self.optimizer.zero_grad()
                outputs2 = self.model(inputs)
                loss2 = self.criterion(outputs2, targets)
                loss2.backward()
                self.optimizer.compute_estimator()
        else:
            if not first_step:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                if self.params['projection_radius'] is not None:
                    with torch.no_grad():
                        for param in self.model.parameters():
                            param.data = proj(param.data, self.params['projection_radius'])

        return outputs, loss

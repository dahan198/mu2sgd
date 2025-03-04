import torch.nn as nn
import torch
import torchmetrics
import os
from .trainer import Trainer
from prettytable import PrettyTable
from tqdm import tqdm
import yaml


class CLFTrainer(Trainer):
    """
    A specialization of the Trainer class for classification tasks.

    This class handles the training, evaluation, and logging of metrics specifically for
    multi-class classification problems. It includes metrics such as loss and accuracy
    and provides functionality to save the best-performing model.

    Attributes:
    - criterion (nn.Module): Loss function used for training, initialized as CrossEntropyLoss.
    - accuracy_metric (torchmetrics.Accuracy): Metric for calculating classification accuracy.
    - best_accuracy (float): Stores the highest test accuracy observed during training.
    - iter_results (dict): Tracks intermediate results for metrics and parameters.
    """

    def __init__(self, model, optimizer, train_dataloader, test_dataloader, params):
        """
        Initializes the CLFTrainer.

        Args:
        - model (nn.Module): The PyTorch model to train.
        - optimizer (torch.optim.Optimizer): Optimizer used for training.
        - train_dataloader (DataLoader): DataLoader for training data.
        - test_dataloader (DataLoader): DataLoader for test/validation data.
        - params (Namespace): Hyperparameters and configurations.
        """
        super().__init__(model, optimizer, train_dataloader, test_dataloader, params)
        self.criterion = nn.CrossEntropyLoss()
        with open(params.task_config_path, 'r') as file:
            self.clf_conf = yaml.safe_load(file)
        self.accuracy_metric = torchmetrics.Accuracy(task="multiclass",
                                                     num_classes=self.clf_conf['classes_num']).to(self.device)
        self.best_accuracy = 0.0
        self.iter_results = {
            "iteration": 0,
            "train_loss": 0,
            "train_acc": 0,
            "test_loss": 0,
            "test_acc": 0,
            **params.__dict__,
        }

    def train(self, epoch_num: int = 50, eval_interval: int = None):
        """
        Trains the model for a specified number of epochs or iterations.

        Args:
        - epoch_num (int, optional): Number of epochs to train. Default is 50.
        - eval_interval (int, optional): Interval for evaluation during training (in iterations). Default is None.
        """
        super().train()
        self.model.train()  # Set the model to training mode
        self.accuracy_metric.reset()

        # Perform the first optimization step
        data, label = next(iter(self.train_dataloader))
        self.make_optimization_step(data, label, first_step=True)

        # Initialize the metrics table
        metrics_table = PrettyTable()
        iter_title = "Epoch" if eval_interval is None else "Iteration"
        metrics_table.field_names = [iter_title, "Train Loss", "Train Accuracy", "Test Loss", "Test Accuracy"]

        total_iterations = (
            int(epoch_num * (len(self.train_dataloader) / eval_interval)) if eval_interval else epoch_num
        )

        running_loss = 0.0
        k = 0

        # Training loop
        for epoch in range(epoch_num):
            if eval_interval is not None:
                for inputs, labels in tqdm(self.train_dataloader):
                    k += 1
                    outputs, loss = self.make_optimization_step(inputs, labels)
                    running_loss += loss.item()
                    predictions = torch.argmax(outputs, dim=1)
                    self.accuracy_metric.update(predictions, labels.to(self.device))

                    if k % eval_interval == 0:
                        self.make_evaluation_step(
                            (k // eval_interval) - 1, total_iterations, eval_interval, running_loss, metrics_table,
                            "Iteration"
                        )
                        running_loss = 0.0
            else:
                running_loss = 0.0
                for inputs, labels in tqdm(self.train_dataloader):
                    outputs, loss = self.make_optimization_step(inputs, labels)
                    running_loss += loss.item()
                    predictions = torch.argmax(outputs, dim=1)
                    self.accuracy_metric.update(predictions, labels.to(self.device))

                self.make_evaluation_step(epoch, epoch_num, len(self.train_dataloader), running_loss, metrics_table,
                                          "Epoch")

        self.save_metrics_and_params()
        print('Finished Training')

    def make_evaluation_step(self, iteration, total_iterations, eval_interval, running_loss, metrics_table, iter_title):
        """
        Performs evaluation during training and logs the results.

        Args:
        - iteration (int): Current training iteration.
        - total_iterations (int): Total number of training iterations.
        - eval_interval (int): Evaluation interval for logging metrics.
        - running_loss (float): Accumulated training loss.
        - metrics_table (PrettyTable): Table for displaying metrics.
        - iter_title (str): Title for the evaluation iteration (e.g., "Epoch" or "Iteration").
        """
        train_accuracy = self.accuracy_metric.compute()
        average_loss = running_loss / eval_interval

        # Evaluate the model on the test dataset
        test_accuracy, test_loss = self.evaluate()

        # Add metrics to the table
        metrics_table.add_row(
            [
                f"{iteration + 1}/{total_iterations}",
                f"{average_loss:.4f}",
                f"{train_accuracy:.2%}",
                f"{test_loss:.4f}",
                f"{test_accuracy:.2%}",
            ]
        )

        # Log the table
        table_string = metrics_table.get_string()
        with open(self.log_file_path, 'a') as log_file:
            log_file.write(f"{table_string}\n\n")

        print(metrics_table)
        metrics_table.clear_rows()

        # Save metrics
        self.metrics_data.append({
            iter_title: iteration,
            "Train Loss": average_loss,
            "Train Accuracy": train_accuracy.item(),
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy.item(),
        })

        # Log metrics to wandb
        if self.use_wandb:
            self.wandb.log({
                "Train Loss": average_loss,
                "Train Accuracy": train_accuracy,
                "Test Loss": test_loss,
                "Test Accuracy": test_accuracy,
            })

        # Save the best-performing model
        if test_accuracy > self.best_accuracy:
            self.best_accuracy = test_accuracy
            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, "best.pth"))

        self.accuracy_metric.reset()

    def evaluate(self):
        """
        Evaluates the model on the test dataset.

        Returns:
        - final_accuracy (float): Accuracy of the model on the test dataset.
        - test_loss (float): Average loss of the model on the test dataset.
        """
        self.model.eval()
        self.accuracy_metric.reset()

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

                predictions = torch.argmax(outputs, dim=1)
                self.accuracy_metric.update(predictions, labels)

        final_accuracy = self.accuracy_metric.compute()
        test_loss = total_loss / total_samples

        self.model.train()
        return final_accuracy, test_loss

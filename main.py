import argparse
from mu2sgd.dataset import DATASET_REGISTRY
from mu2sgd.model import MODEL_REGISTRY
from mu2sgd.trainer import TRAINER_REGISTRY
from mu2sgd.optimizer import OPTIMIZER_REGISTRY
from torch.utils.data import DataLoader
from mu2sgd.utils import set_seed, filter_valid_args
from mu2sgd.utils import get_device


def main():
    """
    The main function for training machine learning models using a configurable pipeline.

    This script allows users to specify various parameters such as the dataset, model architecture,
    optimizer, training epochs, and other hyperparameters through command-line arguments.
    """

    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Training script for various machine learning models.")

    # Define command-line arguments
    parser.add_argument('--task', type=str, default='clf', choices=TRAINER_REGISTRY.keys(),
                        help='Task for the machine learning model (e.g., clf for classification).')
    parser.add_argument('--task_config_path', type=str,
                        default='./config/task/clf.yaml', help='Path to the configuration file for the task.')
    parser.add_argument('--config_folder_path', type=str, default='./config',
                        help='Path to the folder containing configuration files.')
    parser.add_argument('--dataset', type=str, required=True, choices=DATASET_REGISTRY.keys(),
                        help='Dataset to be used for training.')
    parser.add_argument('--model', type=str, required=True, choices=MODEL_REGISTRY.keys(),
                        help='Model architecture to be used.')
    parser.add_argument('--epoch_num', type=int, required=True, help='Number of training epochs.')
    parser.add_argument('--eval_interval', type=int,
                        help='Evaluation interval during training (in iterations).')
    parser.add_argument('--optimizer', type=str, required=True, choices=OPTIMIZER_REGISTRY.keys(),
                        help='Optimizer to be used for training.')
    parser.add_argument('--learning_rate', type=float, required=True,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--gradient_momentum', type=float, default=0.9,
                        help='Momentum for the gradient (if applicable to the optimizer).')
    parser.add_argument('--use_alpha_t', action='store_true',
                        help='Flag to enable the use of alpha=t feature in the optimizer.')
    parser.add_argument('--use_beta_t', action='store_true',
                        help='Flag to enable the use of beta=t feature in the optimizer.')
    parser.add_argument('--query_point_momentum', type=float, default=0.1,
                        help='Momentum for the query point (specific to certain training regimes).')
    parser.add_argument('--batch_size', type=int, default=25, help='Batch size for training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--projection_radius', type=int,
                        help='Projection radius for constraint-based training (if applicable).')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Flag to enable visualization of results with Weights & Biases.')

    # Parse command-line arguments
    args = parser.parse_args()

    # Set the random seed for reproducibility
    set_seed(args.seed)

    # Load the model
    device = get_device()
    model = MODEL_REGISTRY[args.model]().to(device)

    # Load the dataset and create DataLoaders
    dataset = DATASET_REGISTRY[args.dataset]()
    train_dataloader = DataLoader(dataset.trainset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset.testset, batch_size=args.batch_size, shuffle=False)

    # Configure the optimizer based on user input
    if args.optimizer == "momentum":
        optimizer_params = {
            "lr": args.learning_rate,
            "momentum": args.gradient_momentum,
            "dampening": args.gradient_momentum
        }
        optimizer = OPTIMIZER_REGISTRY["sgd"]
    elif args.optimizer == "sgd":
        optimizer_params = {
            "lr": args.learning_rate,
            "momentum": 0.0
        }
        optimizer = OPTIMIZER_REGISTRY["sgd"]
    else:
        optimizer_params = {
            "lr": args.learning_rate,
            "momentum": args.gradient_momentum,
            "gamma": args.query_point_momentum,
            "use_alpha_t": args.use_alpha_t,
            "use_beta_t": args.use_beta_t,
            "projection_radius": args.projection_radius
        }
        optimizer = OPTIMIZER_REGISTRY[args.optimizer]

    # Filter valid optimizer arguments
    optimizer_params = filter_valid_args(optimizer, **optimizer_params)

    # Initialize the optimizer
    optimizer = optimizer(model.parameters(), **optimizer_params)

    # Initialize the trainer
    trainer = TRAINER_REGISTRY[args.task](model, optimizer, train_dataloader, test_dataloader, args)

    # Start training
    trainer.train(epoch_num=args.epoch_num, eval_interval=args.eval_interval)


if __name__ == "__main__":
    main()

import subprocess

if __name__ == "__main__":

    for seed in [1, 2, 3]:
        for batch in [32]:
            for lr in [10, 1, 0.1, 0.01, 0.001, 0.0001]:
                for optimizer in ['mu2sgd', 'momentum', 'sgd', 'storm', 'anytime_sgd']:
                    command = [
                        "python", "main.py",
                        "--task", "clf",
                        "--dataset", "cifar10",
                        "--model", "resnet18",
                        "--epoch_num", "25",
                        "--optimizer", optimizer,
                        "--learning_rate", str(lr),
                        "--gradient_momentum", "0.9",
                        "--query_point_momentum", "0.1",
                        "--batch_size", str(batch),
                        "--seed", str(seed),
                        # "--use_wandb",
                        # "--use_alpha_t",
                    ]
                    subprocess.run(command)

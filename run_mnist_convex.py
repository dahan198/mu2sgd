import subprocess

if __name__ == "__main__":

    for seed in [1, 2, 3]:
        for batch in [64]:
            for lr in [10, 1, 0.1, 0.01, 0.001, 0.0001]:
                for optimizer in ['mu2sgd', 'momentum', 'sgd', 'storm', 'anytime_sgd']:
                    command = [
                        "python", "main.py",
                        "--task", "clf",
                        "--dataset", "mnist",
                        "--model", "logistic_regression",
                        "--epoch_num", "1",
                        "--optimizer", optimizer,
                        "--learning_rate", str(lr),
                        "--gradient_momentum", "0.9",
                        "--batch_size", str(batch),
                        "--seed", str(seed),
                        "--eval_interval", "20",
                        "--projection_radius", "1",
                        # "--use_wandb",
                        "--use_alpha_t",
                        "--use_beta_t",
                    ]
                    subprocess.run(command)

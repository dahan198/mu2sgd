import torch
import numpy as np
import random
import inspect
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def proj(w, rad=1):
    norm_w = torch.norm(w)  # Compute the Euclidean norm of the weights
    if norm_w < rad:
        return w  # No projection needed if norm is within the radius
    else:
        return (w / norm_w) * rad  # Project onto the ball of radius `rad


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def filter_valid_args(object_class, **kwargs):
    init_signature = inspect.signature(object_class.__init__)
    valid_params = set(init_signature.parameters.keys())
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return filtered_kwargs


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_raw_data(results_folder: str):
    # Path to the directory containing all the run folders
    base_path = results_folder
    dfs = []
    # Iterate over each item in the base directory
    for item in os.listdir(base_path):
        # Construct the full path to the item
        item_path = os.path.join(base_path, item)
        # Check if the item is a directory (e.g., a run folder)
        if os.path.isdir(item_path):
            # Path to the results.csv file within the run folder
            results_path = os.path.join(item_path, 'results.csv')
            # Check if the results.csv file exists
            if os.path.exists(results_path):
                # Read the results.csv file
                results_data = pd.read_csv(results_path)
                # Append the DataFrame to the list
                dfs.append(results_data)

    # Concatenate all DataFrames in the list into a single DataFrame
    df = pd.concat(dfs, ignore_index=True)

    return df


def get_averaged_results(raw_df: pd.DataFrame):
    # Remove empty columns (columns with all NaN or no values)
    non_empty_columns = [col for col in raw_df.columns if not raw_df[col].isna().all()]

    # Define columns to group by (all non-empty columns except 'Train Loss', 'Train Accuracy', 'Test Loss',
    # 'Test Accuracy', and 'seed')
    group_columns = [col for col in non_empty_columns if
                     col not in ['Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy', 'seed']]

    # Group the DataFrame by the specified columns
    grouped = raw_df.groupby(group_columns)

    # Calculate the mean and standard deviation for the relevant metrics
    result_mean = grouped[['Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy']].mean().reset_index()
    result_std = grouped[['Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy']].std().reset_index()

    # Merge the mean and std dataframes if needed to align them side by side for better comparison
    result = pd.merge(result_mean, result_std, on=group_columns, suffixes=('_mean', '_std'))
    return result


def plot_test_metric_over_epochs(df: pd.DataFrame,
                                 lrs=[0.0001, 0.001, 0.01, 0.1, 1, 10],
                                 optimizer_order=['mu2sgd', 'momentum', 'sgd', 'storm', 'anytime_sgd'],
                                 y_column='Test Accuracy',
                                 x_column='Epoch',
                                 output_dir="plots"):
    """
    Plots test accuracy over epochs for given optimizers and learning rates.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with columns 'Epoch', 'Test Accuracy_mean', 'Test Accuracy_std',
                         'optimizer', and 'learning_rate'.
    - optimizer_order (list): List of optimizers in the desired plotting order.
    - output_dir (str): Directory to save the plots. Defaults to "plots".

    Returns:
    - None: Saves plots to files and shows them.
    """
    # Ensure the 'optimizer' column is categorical with the specified order
    df['optimizer'] = pd.Categorical(df['optimizer'], categories=optimizer_order, ordered=True)

    # Set Seaborn style and context
    sns.set(style="whitegrid")
    sns.set_context("talk")

    # Define a color palette for the optimizers
    palette = sns.color_palette("coolwarm", n_colors=len(optimizer_order))

    # Create a plot for each learning rate
    for lr in lrs:
        # Filter the DataFrame for the current learning rate
        df_lr = df[df['learning_rate'] == lr]

        # Initialize a subplot grid of 2 rows and 3 columns
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12), sharex=True, sharey=True)

        # Add the title at the bottom
        fig.text(0.5, -0.02, f'{y_column} Over {x_column}s at Learning Rate = {lr}', ha='center', fontsize=16)

        # Plot combined graph on the first subplot
        sns.lineplot(data=df_lr, x=f'{x_column}', y=f'{y_column}_mean', hue='optimizer',
                     palette=palette, ax=axes[0, 0], legend='full')
        axes[0, 0].set_title(f'{y_column}')
        axes[0, 0].set_ylabel(f'{y_column}')

        # Plot for each optimizer in separate subplots
        for i, optimizer in enumerate(optimizer_order, start=1):
            ax = axes.flatten()[i]
            # Filter data for the optimizer
            df_opt = df_lr[df_lr['optimizer'] == optimizer]

            # Plot mean Test Accuracy
            ax.plot(df_opt[f'{x_column}'], df_opt[f'{y_column}_mean'], label=f'{optimizer}', color=palette[i - 1])

            # Add shaded area for std deviation
            ax.fill_between(df_opt[f'{x_column}'],
                            df_opt[f'{y_column}_mean'] - df_opt[f'{y_column}_std'],
                            df_opt[f'{y_column}_mean'] + df_opt[f'{y_column}_std'],
                            color=palette[i - 1], alpha=0.3, label='±1 std dev')
            ax.set_title(f'{optimizer}')
            ax.set_xlabel(f'{x_column} Number')
            ax.set_ylabel(f'{y_column}')

        # Adjust layout and save the figure
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the rect to make room for the suptitle
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/{y_column}_LR_{lr}.png"
        plt.savefig(output_file)
        plt.show()


def plot_metric_vs_learning_rate(df,
                                 lr_range=(0.0001, 10),
                                 x_limits=(0.1, 1),
                                 y_limits=(0.8, 0.9),
                                 h_line=None,
                                 y_column='Test Accuracy',
                                 optimizer_order=['mu2sgd', 'momentum', 'sgd', 'storm', 'anytime_sgd'],
                                 plot_order=['momentum', 'sgd', 'storm', 'anytime_sgd', 'mu2sgd'],
                                 display_std=False,
                                 figsize=(16, 8),
                                 output_dir="plots"):
    """
    Plots Test Accuracy over Learning Rates for various optimizers with standard deviation as shaded areas.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the results with columns 'optimizer', 'learning_rate', 'Test Accuracy_mean', and 'Test Accuracy_std'.
    - lr_range (tuple): Range of learning rates to include in the plot.
    - x_limits (tuple): Limits for the x-axis.
    - y_limits (tuple): Limits for the y-axis.
    - h_line (float): Value for a horizontal dashed green line. If None, no line is plotted.
    - y_column (str): Column name for the y-axis.
    - optimizer_order (list): List of optimizers in the desired plotting order.
    - plot_order (list): The order in which optimizers are plotted.
    - metric (str): Either 'max' (default) or 'min'.
    - last_by (str): Column to sort by for the last iteration.
    - display_std (bool): If True, adds standard deviation as shaded areas.
    - output_dir (str): Directory to save the plot.

    Returns:
    - matplotlib.figure.Figure: The generated plot figure.
    """
    # Set Seaborn style and context
    sns.set(style="whitegrid")
    sns.set_context("talk")

    # Filter the DataFrame based on the learning rate range
    df = df[(df['learning_rate'] <= lr_range[1]) & (df['learning_rate'] >= lr_range[0])]

    # Ensure the 'optimizer' column is categorical with the specified order
    df['optimizer'] = pd.Categorical(df['optimizer'], categories=optimizer_order, ordered=True)

    # Determine the metric to use
    if y_column == 'Test Accuracy':
        # Get the max Test Accuracy_mean for each optimizer and learning rate
        summary = df.loc[df.groupby(['optimizer', 'learning_rate'])[f'{y_column}_mean'].idxmax()]
    else:
        # Get the last iteration for each optimizer and learning rate
        summary = df.loc[df.groupby(['optimizer', 'learning_rate'])[f'{y_column}_mean'].idxmin()]

    # Sorting the DataFrame by optimizer according to the defined order
    summary.sort_values(by='optimizer', inplace=True)

    # Define a color palette
    palette = sns.color_palette("coolwarm", n_colors=len(optimizer_order))
    palette_dict = {optimizer: palette[i] for i, optimizer in enumerate(optimizer_order)}

    # Plotting
    plt.figure(figsize=figsize)

    # Plot each optimizer
    for optimizer in plot_order:
        data = summary[summary['optimizer'] == optimizer]
        sns.lineplot(data=data, x='learning_rate', y=f'{y_column}_mean', label=optimizer, marker='o',
                     color=palette_dict[optimizer])

        if display_std:
            # Add shaded area for std
            plt.fill_between(data['learning_rate'],
                             data[f'{y_column}_mean'] - data[f'{y_column}_std'],
                             data[f'{y_column}_mean'] + data[f'{y_column}_std'],
                             color=palette_dict[optimizer], alpha=0.3)

    # Set x-axis and y-axis limits
    if x_limits:
        plt.xlim(x_limits)
    if y_limits:
        plt.ylim(y_limits)

    # Add horizontal line if specified
    if h_line is not None:
        plt.axhline(y=h_line, color='green', linestyle='--', label=f'{y_column}={h_line}')

    # Set labels and scale
    plt.xlabel('Learning Rate', fontsize=12)
    plt.ylabel(f'{y_column}', fontsize=12)
    plt.xscale('log')  # Learning rates often vary in orders of magnitude
    plt.legend(fontsize=16, loc='lower left')
    plt.grid(True)

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{y_column}_Over_LRs-{lr_range[0]}-{lr_range[1]}.png"

    # Save and show the plot
    plt.savefig(output_file)
    plt.show()

    return plt.gcf()  # Return the current figure


def plot_metric_over_iterations_in_one_plot(df,
                                            optimizer_order=['mu2sgd', 'momentum', 'sgd', 'storm', 'anytime_sgd'],
                                            y_column='Test Loss',
                                            lrs=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                                            x_column='Iteration',
                                            y_limits=None,
                                            figsize=(18, 12),
                                            output_dir="plots"):
    """
    Plots Test Loss over Iterations for various learning rates and optimizers.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the results.
    - optimizer_order (list): List of optimizers in the desired plotting order.
    - learning_rates (list, optional): Specific learning rates to include in the plot. If None, uses all learning rates.
    - y_column (str): Column name for the y-axis (default is 'Test Loss').
    - std_column (str): Column name for the standard deviation (default is 'Test Loss_std').
    - iteration_column (str): Column name for the iterations (default is 'Iteration').
    - palette (list, optional): Custom color palette for the optimizers. If None, a default palette is used.
    - output_file (str): File path to save the plot image (default is 'test_loss_over_iterations.png').

    Returns:
    - None: Saves the plot to the specified file and displays it.
    """
    # Set the style and context for Seaborn plots
    sns.set(style="whitegrid")
    sns.set_context("talk")

    # Define a better color palette if not provided
    palette = sns.color_palette("coolwarm", n_colors=len(optimizer_order))

    # Filter DataFrame for the specified learning rates
    learning_rates = sorted(lrs)

    # Ensure the 'optimizer' column is categorical with the specified order
    df['optimizer'] = pd.Categorical(df['optimizer'], categories=optimizer_order, ordered=True)

    # Create a subplot grid
    num_subplots = len(learning_rates)
    ncols = 4  # Number of columns in the grid
    nrows = (num_subplots + ncols - 1) // ncols  # Calculate rows dynamically
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True)

    # Flatten axes for easy iteration and hide unused subplots
    axes = axes.flatten()
    for ax in axes[num_subplots:]:
        ax.axis('off')

    # Plot for each learning rate
    for i, lr in enumerate(reversed(learning_rates)):
        ax = axes[i]
        # Filter the DataFrame for the current learning rate
        df_lr = df[df['learning_rate'] == lr]

        # Plot the mean Test Loss
        sns.lineplot(data=df_lr, x=x_column, y=f'{y_column}_mean', hue='optimizer',
                     palette=palette, ax=ax, legend='full' if i == 0 else None)

        # Fill the area between the mean ± 1 std deviation
        for j, optimizer in enumerate(optimizer_order):
            df_opt = df_lr[df_lr['optimizer'] == optimizer]
            ax.fill_between(df_opt[x_column],
                            df_opt[f'{y_column}_mean'] - df_opt[f'{y_column}_std'],
                            df_opt[f'{y_column}_mean'] + df_opt[f'{y_column}_std'],
                            color=palette[j], alpha=0.3)

        ax.set_title(f'Learning Rate: {lr}')
        ax.set_ylabel(f'{y_column}')
        ax.set_xlabel(x_column)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the rect to make room for the title

    # Save and show the plot
    output_file = f"{output_dir}/{y_column}_Over_{x_column}_And_LRs.png"

    if y_limits:
        plt.ylim(y_limits)

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_file)
    plt.show()

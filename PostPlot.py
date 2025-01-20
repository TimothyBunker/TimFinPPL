import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


def concatenate_files(folder_path, output_file):
    """
    Concatenates all files in a folder in chronological order.

    Args:
        folder_path (str): Path to the folder containing the files.
        output_file (str): Path to save the concatenated output file.

    Returns:
        pd.DataFrame: The concatenated DataFrame.
    """
    # List all files in the folder and sort them in ascending order
    all_files = sorted(os.listdir(folder_path))
    print(f"Processing all files: {all_files}")

    # Read and concatenate all files
    combined_df = pd.DataFrame()
    for file in all_files:
        file_path = os.path.join(folder_path, file)
        if file.endswith(".csv"):  # Adjust for your file format
            df = pd.read_csv(file_path)
        elif file.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        else:
            print(f"Skipping unsupported file format: {file}")
            continue
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Save the concatenated DataFrame to the output file
    if output_file.endswith(".csv"):
        combined_df.to_csv(output_file, index=False)
    elif output_file.endswith(".parquet"):
        combined_df.to_parquet(output_file, index=False)
    else:
        raise ValueError("Output file must be a CSV or Parquet file.")

    print(f"Concatenated file saved to {output_file}")
    return combined_df

def plot_training_losses_with_trend(df, degree=1):
    """
    Plots training losses with a trend line.

    Args:
        df (pd.DataFrame): DataFrame containing 'actor_loss', 'critic_loss', and 'total_loss'.
        degree (int): Degree of the polynomial for the trend line (1 = linear).
    """
    plt.figure(figsize=(10, 6))

    # Extract x-axis (iteration) and y-axis (losses)
    x = np.arange(len(df))  # Iterations as x-axis
    actor_loss = df['actor_loss']
    critic_loss = df['critic_loss']
    total_loss = df['total_loss']

    # Fit polynomial trend lines
    actor_trend = np.poly1d(np.polyfit(x, actor_loss, degree))
    critic_trend = np.poly1d(np.polyfit(x, critic_loss, degree))
    total_trend = np.poly1d(np.polyfit(x, total_loss, degree))

    # # Plot the raw losses
    # plt.plot(x, actor_loss, label='Actor Loss', color='blue', alpha=0.5)
    # plt.plot(x, critic_loss, label='Critic Loss', color='orange', alpha=0.5)
    # plt.plot(x, total_loss, label='Total Loss', color='green', alpha=0.5)

    # Plot the trend lines
    plt.plot(x, actor_trend(x), label='Actor Loss Trend', color='blue', linewidth=2)
    plt.plot(x, critic_trend(x), label='Critic Loss Trend', color='orange', linewidth=2)
    plt.plot(x, total_trend(x), label='Total Loss Trend', color='green', linewidth=2)

    # Customize axes
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Training Losses with Trend Line (Degree: {degree})')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add gridlines

    plt.show()

def plot_training_losses(df):
    """
    Plots the training losses with customized axis scaling.

    Args:
        df (pd.DataFrame): DataFrame containing 'actor_loss', 'critic_loss', and 'total_loss'.
    """
    plt.figure(figsize=(10, 6))

    # Plot the losses
    plt.plot(df['actor_loss'], label='Actor Loss')
    plt.plot(df['critic_loss'], label='Critic Loss')
    plt.plot(df['total_loss'], label='Total Loss')

    # Customize axes
    y_min, y_max = df[['actor_loss', 'critic_loss', 'total_loss']].min().min(), df[
        ['actor_loss', 'critic_loss', 'total_loss']].max().max()
    padding = (y_max - y_min) * 0.1  # Add 10% padding
    plt.ylim(y_min - padding, y_max + padding)  # Set y-axis range dynamically

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Losses with Custom Axis Scaling')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add gridlines

    plt.show()


def scale_data(df, method="min-max"):
    """
    Scales the data in the DataFrame using the specified method.

    Args:
        df (pd.DataFrame): DataFrame containing columns to scale.
        method (str): Scaling method ('min-max', 'z-score', or 'log').

    Returns:
        pd.DataFrame: Scaled DataFrame.
    """
    scaled_df = df.copy()
    if method == "min-max":
        for column in ['actor_loss', 'critic_loss', 'total_loss']:
            min_val = df[column].min()
            max_val = df[column].max()
            scaled_df[column] = (df[column] - min_val) / (max_val - min_val)
    elif method == "z-score":
        for column in ['actor_loss', 'critic_loss', 'total_loss']:
            mean = df[column].mean()
            std = df[column].std()
            scaled_df[column] = (df[column] - mean) / (std + 1e-8)
    elif method == "log":
        for column in ['actor_loss', 'critic_loss', 'total_loss']:
            scaled_df[column] = np.log(df[column] + 1e-8)
    else:
        raise ValueError("Unsupported scaling method. Use 'min-max', 'z-score', or 'log'.")
    return scaled_df


if __name__ == '__main__':
    folder_path = "/logs/training"
    output_file = "../concatenated_training_log_1.csv"

    # Concatenate all files in the directory
    concatenated_df = concatenate_files(folder_path, output_file)

    # Plot training losses from the concatenated DataFrame
    # plot_training_losses(concatenated_df)
    plot_training_losses_with_trend(concatenated_df, degree=1)

import os

import pandas as pd
import matplotlib.pyplot as plt

LOGS_DIR = "lightning_logs"


def visualize_one(log_dir: str) -> None:
    version = log_dir.split("_")[-1]

    metrics_csv_path = os.path.join(LOGS_DIR, log_dir, "metrics.csv")

    metrics_df = pd.read_csv(metrics_csv_path)

    selected_columns = [
        "step",
        "train/rec_loss_step",
        "val/rec_loss",
    ]

    plt.figure(figsize=(10, 6))
    for column in selected_columns[1:]:
        cleaned_metrics = metrics_df.dropna(subset=[column])
        plt.plot(cleaned_metrics["step"], cleaned_metrics[column], label=column)
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Losses Over Steps")
    plt.legend()
    plt.grid(which="both", axis="both")
    plt.savefig(f"training_losses_{version}.png")


if __name__ == "__main__":
    logs_dir = os.listdir(LOGS_DIR)
    for log_dir in logs_dir:
        visualize_one(log_dir)

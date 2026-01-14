import os

import pandas as pd
import matplotlib.pyplot as plt

lightning_logs_dir = "lightning_logs"

latest_log_dir = sorted(
    [d for d in os.listdir(lightning_logs_dir) if d.startswith("version_")],
    key=lambda x: int(x.split("_")[-1])
)[-1]

metrics_csv_path = os.path.join(lightning_logs_dir, latest_log_dir, "metrics.csv")

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
plt.grid()
plt.savefig("training_losses.png")
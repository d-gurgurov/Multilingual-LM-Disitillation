import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Set publication-quality styles
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
mpl.rcParams["legend.fontsize"] = 12
mpl.rcParams["lines.markersize"] = 6

# Adapter sizes and F1-scores for different configurations
data = {
    "mBERT": {
        "Original": {
            "R": [2, 4, 8, 16, 32, 64, 128],
            "Params": [7091712, 3550464, 1779840, 894528, 451872, 230544, 119880],
            "F1": [83.60, 82.23, 83.13, 84.91, 83.13, 84.17, 84.38],
        },
        "Hidden-312": {
            "R": [2, 4, 8, 16, 32],
            "Params": [586872, 294372, 148122, 73122, 35622],
            "F1": [73.14, 73.71, 71.72, 71.47, 69.22],
        },
        "Hidden-456": {
            "R": [2, 4, 8, 16, 32],
            "Params": [1251720, 627228, 314982, 156120, 79428],
            "F1": [80.23, 79.88, 77.97, 77.84, 76.84],
        },
        "Hidden-564": {
            "R": [2, 4, 8, 16, 32],
            "Params": [1913652, 958518, 477564, 240474, 118542],
            "F1": [79.47, 81.18, 81.36, 81.51, 80.28],
        },
    },
    "XLM-R": {
        "Original": {
            "R": [2, 4, 8, 16, 32, 64, 128],
            "Params": [7091712, 3550464, 1779840, 894528, 451872, 230544, 119880],
            "F1": [84.76, 84.03, 84.76, 84.96, 84.57, 83.77, 84.32],
        },
        "Hidden-312": {
            "R": [2, 4, 8, 16, 32],
            "Params": [586872, 294372, 148122, 73122, 35622],
            "F1": [66.94, 64.62, 65.86, 65.55, 60.25],
        },
        "Hidden-456": {
            "R": [2, 4, 8, 16, 32],
            "Params": [1251720, 627228, 314982, 156120, 79428],
            "F1": [78.48, 76.39, 74.67, 75.87, 74.32],
        },
        "Hidden-564": {
            "R": [2, 4, 8, 16, 32],
            "Params": [1913652, 958518, 477564, 240474, 118542],
            "F1": [83.43, 83.14, 82.21, 82.18, 81.65],
        },
    },
}

# Modified plotting function to include R values as text annotations above each point

def plot_adapter_results_with_ranks(model_name, model_data):
    plt.figure(figsize=(8, 6))
    for label, values in model_data.items():
        x = values["Params"]
        y = values["F1"]
        r_vals = values["R"]
        plt.plot(x, y, marker='o', label=label)
        for i, (x_val, y_val, r_val) in enumerate(zip(x, y, r_vals)):
            plt.text(x_val, y_val + 0.3, f"R={r_val}", ha="center", fontsize=10)

    plt.title(f"{model_name} - Adapter Size vs F1 (Topic Classification)")
    plt.xlabel("Adapter Size (Parameters)")
    plt.ylabel("F1 Score")
    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{model_name}_r")
    plt.show()

# Generate updated plots
for model, model_data in data.items():
    plot_adapter_results_with_ranks(model, model_data)


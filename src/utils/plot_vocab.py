import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Set publication-quality styles
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["axes.labelsize"] = 14  # Larger labels
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
mpl.rcParams["legend.fontsize"] = 12
mpl.rcParams["lines.markersize"] = 6

# Data
vocab_sizes = [40000, 20000, 15000, 10000, 5000] 
f1_scores_sk = [0.8229, 0.8177, 0.8189, 0.8048, 0.7524]
original_f1_sk = 0.8199

f1_scores_mt = [0.7127, 0.6919, 0.7113, 0.6969, 0.6550]
original_f1_mt = 0.7147

f1_scores_sw = [0.7820, 0.7643, 0.7670, 0.7594, 0.7486]
original_f1_sw = 0.7825

# Plot
plt.figure(figsize=(8, 6)) # yerr=std_devs, 
plt.errorbar(vocab_sizes, f1_scores_sw, fmt="o-", capsize=4, label="Compressed Models")
plt.axhline(original_f1_sw, linestyle="--", color="red", label="Original Vocab Model")
plt.grid(linestyle="--", alpha=0.5)

# Labels and legend
# plt.xlabel("Vocabulary Size")
# plt.ylabel("Average F1 Score")
plt.legend()
# plt.xscale("log")  # Optional: Log scale for better spacing
plt.xticks(vocab_sizes, labels=[f"{x/1000:.0f}k" for x in vocab_sizes])  # Format as '40k', '20k', etc.
plt.title("Vocabulary Size for Swahili")
plt.savefig("vocab_sw.png", dpi=300, bbox_inches="tight")
# Show plot
plt.show()
"""
notebooks/visualization.py
Visualization placeholders (use matplotlib in real notebooks).
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt

def plot_latency_hist(latencies):
    plt.hist(latencies, bins=50)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Count")
    plt.title("Latency distribution")
    plt.show()

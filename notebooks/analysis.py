"""
notebooks/analysis.py
Simple analysis utilities for evaluation outputs.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt

def load_results(path):
    with open(path, "r") as f:
        return json.load(f)

def plot_metric(results, metric="recall@10"):
    # expects results from scripts/evaluate
    hybrid = results.get("hybrid", {})
    baselines = {k: v for k, v in results.items() if k not in ["hybrid","ablation"]}
    print("Hybrid", hybrid.get(metric))
    for bname, data in baselines.items():
        print(bname, data.get(metric))

# This script is a placeholder for interactive exploration in notebooks.

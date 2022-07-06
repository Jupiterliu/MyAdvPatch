
from utils.evaluation import *
import numpy as np
import matplotlib.pyplot as plt


def plot_metrics_results(path):
    metrics_result_path = os.path.join(path, "metrics_results.txt")
    metrics_results = np.loadtxt(metrics_result_path)
    plt.figure(figsize=(20, 5))
    min = np.min(metrics_results[:, 0])
    max = np.max(metrics_results[:, 0])
    x_ticks = np.arange(min, max + 0.1, 0.2)
    plt.xticks(x_ticks)

    plt.subplot(1, 2, 1)
    plt.title("Steering Angle")
    l1 = plt.plot(metrics_results[:, 0], metrics_results[:, 1], 'r--', label='ASD')
    l2 = plt.plot(metrics_results[:, 0], metrics_results[:, 2], 'b--', label='MAE')
    l3 = plt.plot(metrics_results[:, 0], metrics_results[:, 3], 'g--', label='RMSE')
    plt.plot(metrics_results[:, 0], metrics_results[:, 1], "ro-",
             metrics_results[:, 0], metrics_results[:, 2], "bv-",
             metrics_results[:, 0], metrics_results[:, 3], "g^-")
    plt.xlabel("Scale")
    plt.ylabel("Errors")
    plt.xticks(x_ticks)
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Collision Prob.")
    l4 = plt.plot(metrics_results[:, 0], metrics_results[:, 4], 'c--', label='mASR')
    l5 = plt.plot(metrics_results[:, 0], metrics_results[:, 5], 'y--', label='mF1-score')
    plt.plot(metrics_results[:, 0], metrics_results[:, 4], "cs-",
             metrics_results[:, 0], metrics_results[:, 5], "yD-")
    plt.xlabel("Scale")
    plt.ylabel("ASR")
    plt.xticks(x_ticks)
    plt.grid()
    plt.legend()

    plt.savefig(os.path.join(path, 'metrics_results.png'))
    plt.tight_layout()
    plt.show()
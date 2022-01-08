import matplotlib

matplotlib.use('TkAgg')

from typing import List
from AdvancedDL import LOGS_DIR

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

text_size = 12
axes_font_size = 12
linewidth = 1
markersize = 5
labelsize = 12
plt.rcParams["figure.figsize"] = [16, 9]
plt.rcParams['axes.labelsize'] = axes_font_size
plt.rcParams['axes.titlesize'] = text_size
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['lines.linewidth'] = linewidth
plt.rcParams['lines.markersize'] = markersize
plt.rcParams['xtick.labelsize'] = labelsize
plt.rcParams['ytick.labelsize'] = labelsize
plt.rcParams['font.size'] = text_size
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['legend.fontsize'] = text_size

colors = ['b', 'r', 'b', 'r']
markers = ['x', 'o', 'x', 'o']


def extract_results(file_path: str) -> List[float]:
    with open(file_path, 'rb') as f:
        results = pickle.load(f)
        results = results[0]

    return results


exp_name = "MoCoV2_ResNet50_1000_STE_250_E_2022-01-05"
save_dir = os.path.join(LOGS_DIR, exp_name, 'Plots')
os.makedirs(save_dir, exist_ok=True)

self_training_dir = os.path.join(LOGS_DIR, exp_name, 'SelfSupervisedLearning')
supervised_training_dir = os.path.join(LOGS_DIR, exp_name, 'SupervisedLearning')
legend = [
    "Train",
    "Test",
]

self_training_training_loss_file = os.path.join(self_training_dir, 'Train', 'fit_train_loss_1.pkl')
self_training_training_acc_file = os.path.join(self_training_dir, 'Train', 'fit_train_acc_1.pkl')
self_training_eval_loss_file = os.path.join(self_training_dir, 'Val', 'fit_eval_loss_1.pkl')
self_training_eval_acc_file = os.path.join(self_training_dir, 'Val', 'fit_eval_acc_1.pkl')

self_training_training_loss = extract_results(self_training_training_loss_file)
self_training_training_acc = extract_results(self_training_training_acc_file)
self_training_eval_loss = extract_results(self_training_eval_loss_file)
self_training_eval_acc = extract_results(self_training_eval_acc_file)

supervised_training_training_loss_file = os.path.join(supervised_training_dir, 'Train', 'fit_train_loss_1.pkl')
supervised_training_training_acc_file = os.path.join(supervised_training_dir, 'Train', 'fit_train_acc_1.pkl')
supervised_training_test_loss_file = os.path.join(supervised_training_dir, 'Test', 'fit_eval_loss_1.pkl')
supervised_training_test_acc_file = os.path.join(supervised_training_dir, 'Test', 'fit_eval_acc_1.pkl')

supervised_training_training_loss = extract_results(supervised_training_training_loss_file)
supervised_training_training_acc = extract_results(supervised_training_training_acc_file)
supervised_training_test_loss = extract_results(supervised_training_test_loss_file)
supervised_training_test_acc = extract_results(supervised_training_test_acc_file)

self_supervised_training_x_axis = np.arange(len(self_training_training_loss))
supervised_training_x_axis = np.arange(len(supervised_training_training_loss))

fig, axes = plt.subplots(nrows=2)
axes[0].plot(
    self_supervised_training_x_axis,
    self_training_training_loss,
    c=colors[0],
    marker=markers[0],
    markersize=markersize,
    fillstyle='none',
)
axes[0].plot(
    self_supervised_training_x_axis,
    self_training_eval_loss,
    c=colors[1],
    marker=markers[1],
    markersize=markersize,
    fillstyle='none',
)
axes[0].set_ylabel('Cross Entropy Loss')
axes[0].legend(legend)
axes[0].set_title('Self-Supervised Stage')

axes[1].plot(
    supervised_training_x_axis,
    supervised_training_training_loss,
    c=colors[2],
    marker=markers[2],
    markersize=markersize,
    fillstyle='none',
)
axes[1].plot(
    supervised_training_x_axis,
    supervised_training_test_loss,
    c=colors[3],
    marker=markers[3],
    markersize=markersize,
    fillstyle='none',
)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Cross Entropy Loss')
axes[1].set_title('Supervised Stage')

plt.savefig(
    fname=os.path.join(save_dir, 'Loss.png'),
    orientation='landscape',
    format='png',
)

fig, axes = plt.subplots(nrows=2)
axes[0].plot(
    self_supervised_training_x_axis,
    self_training_training_acc,
    c=colors[0],
    marker=markers[0],
    markersize=markersize,
    fillstyle='none',
)
axes[0].plot(
    self_supervised_training_x_axis,
    self_training_eval_acc,
    c=colors[1],
    marker=markers[1],
    markersize=markersize,
    fillstyle='none',
)
axes[0].set_ylabel('Binary Accuracy (Augmented Vs. Memory Bank)')
axes[0].legend(legend)
axes[0].set_title('Self-Supervised Stage')

axes[1].plot(
    supervised_training_x_axis,
    supervised_training_training_acc,
    c=colors[2],
    marker=markers[2],
    markersize=markersize,
    fillstyle='none',
)
axes[1].plot(
    supervised_training_x_axis,
    supervised_training_test_acc,
    c=colors[3],
    marker=markers[3],
    markersize=markersize,
    fillstyle='none',
)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Top-1 Accuracy')
axes[1].set_title('Supervised Stage')

plt.savefig(
    fname=os.path.join(save_dir, 'Accuracy.png'),
    orientation='landscape',
    format='png',
)

plt.show()

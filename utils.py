import torch.nn as nn
import torch


def accuracy_correct(pred: torch.Tensor, label: torch.Tensor):
    pred_label = pred.argmax(dim=1)
    correct = (pred_label == label).sum().item()
    # accuracy = correct / label.size(0)
    return correct, label.size(0)

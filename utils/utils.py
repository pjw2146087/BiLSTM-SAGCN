import torch
import math
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import glob
import os
import torch
import torch.nn as nn
class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_start_lr, end_lr, warmup_epochs, total_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_start_lr = warmup_start_lr
        self.end_lr = end_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # linear warmup
        if self.last_epoch < self.warmup_epochs:
            lr = (self.base_lrs[0] - self.warmup_start_lr) * float(self.last_epoch) / float(self.warmup_epochs) + self.warmup_start_lr
            return [lr]

        # cosine annealing decay
        progress = float(self.last_epoch - self.warmup_epochs) / float(max(1, self.total_epochs - self.warmup_epochs))
        cosine_lr = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        lr = max(0.0, cosine_lr * (self.base_lrs[0] - self.end_lr) + self.end_lr)
        return [lr]


def label_smoothing_loss(pred, label, weight, epsilon=0.1):
    n_class = pred.shape[1]
    one_hot = torch.nn.functional.one_hot(label.view(-1), n_class).float()
    smoothed_one_hot = (1.0 - epsilon) * one_hot + epsilon / n_class
    loss = torch.nn.functional.cross_entropy(pred.float(), smoothed_one_hot, weight=weight, reduction='mean')
    return loss
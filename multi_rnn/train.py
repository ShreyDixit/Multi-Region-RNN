from copy import deepcopy
from typing import Callable
import torch
import torch.nn as nn
from fastprogress.fastprogress import master_bar, progress_bar
import numpy as np

def train(model: nn.Module, dataset: Callable, optimizer: torch.optim.Optimizer, criterion: nn.Module, iterations: int, device: str, plot_loss: bool = False):
    if plot_loss:
        pb = master_bar(range(1, iterations+1), )
    else:
        pb = progress_bar(range(1, iterations+1))

    train_loss = []
    model_best = model
    best_loss = np.inf
    for i in pb:
        inputs, labels = dataset()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)

        loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

        if loss < best_loss:
            best_loss = loss
            model_best = deepcopy(model)

        if plot_loss:
            plot_loss_update(i, iterations, pb, train_loss)

        pb.comment = f"loss: {loss.item():.5f}"

    return model_best



def plot_loss_update(epoch, epochs, pb, train_loss):
    """ dynamically print the loss plot during the training/validation loop.
        expects epoch to start from 1.
    """
    x = range(1, epoch+1)
    y = np.array(train_loss)
    graphs = [[x,train_loss]]
    x_margin = 1
    y_margin = 0.05
    x_bounds = [1-x_margin, epochs+x_margin]
    y_bounds = [np.min(y)-y_margin, np.max(y)+y_margin]

    pb.update_graph(graphs, x_bounds, y_bounds)
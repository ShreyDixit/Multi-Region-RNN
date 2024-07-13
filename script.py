
import numpy as np
import torch
import torch.nn as nn

import neurogym as ngym
import pickle

import matplotlib.pyplot as plt
from multi_rnn.model import MiltiRegionRNN
from multi_rnn.train import train
from multi_rnn.tasks import DataSet
from multi_rnn.eval import *

from msapy import msa
from functools import cache
import pandas as pd
from multi_rnn.msa import MSA

# %%
task_dict = {
    "PerceptualDecisionMaking-v0": {
        "dt": 100
    },

    "DelayMatchSample-v0": {
        "dt": 100,
    },

    "IntervalDiscrimination-v0": {
        "dt": 100
    },

    "ReadySetGo-v0": {
        "dt": 100
    }
}

seq_len = 100
bs = 128
device = "cuda:2"
hidden_size = 64

dataset = DataSet(seq_len, bs, device, task_dict)


# %%
with open("subgraph_data.pkl", "rb") as f:
    subgraph_data = pickle.load(f, encoding='latin1')
connectome = torch.tensor(subgraph_data['fln_mat']).to(device).float()

# %%
net = MiltiRegionRNN(input_size=4, hidden_size=hidden_size, output_size=3, connectome=connectome).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=3e-5)

# %%
#model = train(net, dataset, optimizer, criterion, 12000, device, plot_loss=False)

# %%
#torch.save(net.state_dict(), f"model_{'_'.join(task_dict.keys())}.pt")
net.load_state_dict(torch.load(f"model_{'_'.join(task_dict.keys())}.pt"))

# %%
print(get_task_wise_accuracy(dataset, 10, net))

# %%
msa = MSA(net, dataset, 2, 500, list(range(29)))
network_interaction = msa.run_network_interaction()
df = pd.DataFrame(network_interaction, index=list(subgraph_data["areas"]), columns=list(subgraph_data["areas"]))
df.to_csv("network_interaction.csv")
# %%




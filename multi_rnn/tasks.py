import neurogym as ngym
import numpy as np
import torch


class DataSet:
    def __init__(self, seq_len, bs, device, task_dicts):
        tasks = list(task_dicts.keys())
        self.tasks = tasks
        self.seq_len = seq_len
        self.bs = bs
        self.device = device
        self.datasets = [ngym.Dataset(task, env_kwargs=task_dicts[task], batch_size=bs, seq_len=seq_len) for task in tasks]
    
    def __call__(self):
        inputs, labels = [], []
        for i, dataset in enumerate(self.datasets):
            input, label = dataset()
            if len(self.datasets) > 1:
                input = np.concatenate((input, np.full((self.seq_len, self.bs, 1), i)), axis=-1)
            inputs.append(input)
            labels.append(label)

        inputs = np.concatenate(inputs, axis=1)
        labels = np.concatenate(labels, axis=1)

        return torch.from_numpy(inputs).type(torch.float).to(self.device), torch.from_numpy(labels).type(torch.long).to(self.device)

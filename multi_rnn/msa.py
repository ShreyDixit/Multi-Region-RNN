from functools import cache
from msapy import msa
import numpy as np
import torch

class MSA:
    def __init__(self, model, dataset, num_batches, num_permutations, elements):
        self.model = model
        self.dataset = dataset
        self.num_batches = num_batches
        self.num_permutations = num_permutations
        self.elements = elements
        self.num_tasks = len(dataset.tasks)
        self.bs = dataset.bs

        self.inputs, self.labels = [], []
        for _ in range(num_batches):
            inputs, labels = dataset()
            self.inputs.append(inputs)
            self.labels.append(labels)

    def objective_function_task_wise(self, complements):
        task_wise_accuracy = [0] * self.num_tasks

        for inputs, labels in zip(self.inputs, self.labels):
            for i in range(self.num_tasks):
                inputs_task = inputs[:, i*self.bs:(i+1)*self.bs]
                labels_task = labels[:, i*self.bs:(i+1)*self.bs].detach().cpu().numpy()

                with torch.no_grad():
                    action_pred = self.model.lesion_forward(inputs_task, complements)
                action_pred = action_pred.cpu().detach().numpy()
                action_pred = np.argmax(action_pred, axis=-1)

                mask = labels_task != 0
                task_wise_accuracy[i] += np.sum(action_pred[mask] == labels_task[mask]) / np.sum(mask)

        return np.array(task_wise_accuracy) / self.num_batches
    
    def objective_function_overall(self, complements):
        return np.mean(self.objective_function_task_wise(complements))
    
    def run_msa(self):
        return msa.interface(
            n_permutations=self.num_permutations,
            objective_function= cache(self.objective_function_task_wise),
            elements=self.elements
        )

    def run_network_interaction(self):
        return msa.network_interaction_2d(
            n_permutations=self.num_permutations,
            objective_function= cache(self.objective_function_overall),
            elements=self.elements,
            random_seed = 2810,
            lazy=True
        )
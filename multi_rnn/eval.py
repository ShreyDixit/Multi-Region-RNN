from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_task_accuracy(dataset, task_bs, task_idx, num_batches, model):
    perf = 0
    for _ in range(num_batches):
        inputs, labels, action_pred = get_preds_labels_inputs(dataset, task_bs, task_idx, model)
        mask = labels != 0
        perf += np.sum(action_pred[mask] == labels[mask]) / np.sum(mask)

    return perf / num_batches

def get_preds_labels_inputs(dataset, task_bs, task_idx, model):
    inputs, labels = get_inputs_labels(dataset, task_bs, task_idx)

    action_pred = model(inputs)
    action_pred = action_pred.cpu().detach().numpy()
    action_pred = np.argmax(action_pred, axis=-1)
    return inputs.detach().cpu().numpy(), labels, action_pred

def get_inputs_labels(dataset, task_bs, task_idx = None):
    inputs, labels = dataset()
    inputs = inputs[:, task_idx*task_bs:(task_idx+1)*task_bs] if task_idx is not None else inputs
    labels = labels[:, task_idx*task_bs:(task_idx+1)*task_bs].detach().cpu().numpy() if task_idx is not None else labels.detach().cpu().numpy()
    return inputs,labels


def get_overall_accuracy(dataset, num_batches, model):
    perf = []
    for i in range(len(dataset.tasks)):
        perf.append(get_task_accuracy(dataset, dataset.bs, i, num_batches, model))
    return np.mean(perf)

def get_task_wise_accuracy(dataset, num_batches, model):
    perf = {}
    for i in range(len(dataset.tasks)):
        perf[dataset.tasks[i]] = get_task_accuracy(dataset, dataset.bs, i, num_batches, model)
    return perf

def plot_task_random_examples(dataset, task_bs, task_idx, model, num_examples = 4, title = "", y_label_inputs = "Task"):
    inputs, labels, action_pred = get_preds_labels_inputs(dataset, task_bs, task_idx, model)

    example_idx = np.random.choice(inputs.shape[1], num_examples)
    n_rows, n_cols = int(np.sqrt(num_examples)), int(np.sqrt(num_examples))

    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(15, 5 * n_rows))

    # add title
    fig.suptitle(title)

    plot_examples(y_label_inputs, inputs, labels, action_pred, example_idx, n_cols, axes)

    plt.tight_layout()
    plt.show()

def plot_example_from_each_task(dataset, task_bs, model, title = "", y_label_inputs = "Task"):
    inputs, labels, action_pred = get_preds_labels_inputs(dataset, task_bs, None, model)
    print(inputs.shape)

    num_tasks = len(dataset.tasks)
    example_idx = [i * task_bs for i in range(num_tasks)]

    n_rows, n_cols = int(np.sqrt(num_tasks)), int(np.sqrt(num_tasks))

    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(15, 5 * n_rows))
    fig.set_dpi(300)
    
    fig.suptitle(title)

    plot_examples(y_label_inputs, inputs, labels, action_pred, example_idx, n_cols, axes)

    plt.tight_layout()
    plt.show()


def plot_examples(y_label_inputs, inputs, labels, action_pred, example_idx, n_cols, axes):
    for i, idx in enumerate(example_idx):
        row = i // n_cols
        col = i % n_cols

        # Plot the inputs as a heatmap
        ax_input = axes[row * 2, col]
        sns.heatmap(inputs[:, idx, :].T, ax=ax_input, cmap="viridis", cbar=False)
        ax_input
        ax_input.set_title(f'Inputs', fontsize=12)
        ax_input.set_ylabel(y_label_inputs)
        ax_input.set_xticks([])
        ax_input.set_yticks([])

        # Plot the labels and action_pred
        ax_output = axes[row * 2 + 1, col]
        ax_output.plot(labels[:, idx], label='Labels', alpha=0.5, linewidth=2)
        ax_output.plot(action_pred[:, idx], label='Predictions', linestyle='--', linewidth=3, alpha=0.5)
        ax_output.set_title(f'Labels and Predictions', fontsize=12)
        ax_output.set_ylabel('Action')
        ax_output.set_xlabel('Time')
        ax_output.set_yticks([])
        ax_output.legend()

def get_task_wise_lesioned_accuracy(dataset, num_batches, model, region_idx):
    perf = []
    lesioned_model_forward = partial(model.lesion_forward, region_idx = region_idx)
    for i in range(len(dataset.tasks)):
        perf.append(get_task_accuracy(dataset, dataset.bs, i, num_batches, lesioned_model_forward))
    
    return np.array(perf)
    
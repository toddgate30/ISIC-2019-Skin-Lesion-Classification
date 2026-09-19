import torch
from torch import nn
import numpy as np

def div_bs(metabatch_images, metabatch_labels, ratio, context):
    """Select a random subset of a meta-batch using Diversified Batch Selection (DivBS).
    
        Args:
            metabatch_images: Images contained in the meta-batch.
            metabatch_labels: Labels corresponding to the meta-batch images.
            ratio: Fraction of the meta-batch to select.
            context: object containing, among other things, the model
    
        Returns:
            A tuple containing the selected images and their corresponding labels.
        """
    model = context.model
    batch_size = len(metabatch_labels) * ratio

    grad_mean, gradients = compute_gradients(metabatch_images, metabatch_labels, model)
    indices = greedy_selection(grad_mean, gradients, batch_size)

    return metabatch_images[indices], metabatch_labels[indices]

def compute_gradients(inputs, targets, model):
    # Saves model's current training so that batch selection does not change it
    model.eval()
    features = None

    def save_features(module, inputs):
        nonlocal features
        features = inputs[0]

    hook = model.fc.register_forward_pre_hook(save_features)

    # try/finally block to make sure hook gets removed.
    try:
        outputs = model(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)
        grad_out = torch.autograd.grad(loss, outputs)[0]

    finally:
        hook.remove()

    gradients = grad_out.unsqueeze(-1) * features.unsqueeze(1)
    gradients = gradients.flatten(start_dim=1)
    grad_mean = gradients.mean(dim=0)

    return grad_mean.detach(), gradients.detach()


def greedy_selection(grad_mean, gradients, number_to_select):
    """Select samples using greedy orthogonalization"""
    residual = grad_mean
    selected_indices = []
    selected_vectors = []

    D = gradients.T

    for _ in range(number_to_select):
        correlations = torch.abs(D.T @ residual)

        if correlations.sum() == 0:
            break

        idx = torch.multinomial(correlations, num_samples=1).item()
        selected_indices.append(idx)

        if selected_vectors:
            selected_matrix = torch.cat(selected_vectors, dim=1)
            selected_vector = (D[:, idx] - selected_matrix @ (selected_matrix.T @ D[:, idx]))
        else:
            selected_vector = D[:, idx]

        norm = torch.norm(selected_vector)
        if norm == 0:
            break

        selected_vector = selected_vector / norm
        selected_vectors.append(selected_vector.unsqueeze(1))

        residual = (residual - (selected_vector @ residual) * selected_vector)

        # If the greedy procedure doesn't obtain enough samples, fill the reamined randomly
        if len(selected_indices) < number_to_select:
            remaining = list(set(range(gradients.shape[0])) - set(selected_indices))
            num_random = number_to_select - len(selected_indices)

            selected_indices.extend(np.random.choice(remaining, num_random, replace=False).tolist())
    return selected_indices
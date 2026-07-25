import torch
import torch.nn as nn
import numpy as np
from scipy import optimize

def forward_pass(context, dataloader):
    model = context.model
    model.eval()
    with torch.no_grad():
        all_logits = []
        all_labels = []
        batches = 0
        for images, labels in dataloader:
            batches += 1
            images = images.to(context.device)
            labels = labels.to(context.device)
            all_logits.append(model(images).cpu())
            all_labels.append(labels)
        logits = torch.cat(all_logits, dim=0)
        predictions = logits.argmax(dim=1)
        log_probs = torch.log_softmax(logits, dim=1)
        labels = torch.cat(all_labels, dim=0)
    return logits, predictions, log_probs, labels

def eval_loss(context, logits, labels):
    return context.loss_function(logits, labels)

def eval_acc(predictions, labels):
    return (predictions == labels).float().mean()

def eval_balanced_acc(predictions, labels):
    num_classes = int(labels.max().item()) + 1
    class_accuracies = []

    for class_idx in range(num_classes):
        class_mask = (labels == class_idx)
        if class_mask.any():
            class_acc = (predictions[class_mask] == labels[class_mask]).float().mean()
        class_accuracies.append(class_acc)
    return torch.stack(class_accuracies).mean()

def eval_progress(log_probs, labels):
    probs = torch.exp(log_probs.detach()).cpu()
    probs = probs / probs.sum(dim=1, keepdim=True)

    probs = probs.numpy().astype(np.float64)
    labels = labels.detach().cpu().numpy().reshape(-1)

    num_samples, num_classes = probs.shape

    predictions = np.sqrt(probs)

    ground_truth = np.zeros_like(probs)
    ground_truth[np.arange(num_samples), labels] = 1.0

    ignorance = np.sqrt(np.full_like(probs, 1.0 / num_classes))
    ignorance_to_truth = np.arccos((ignorance * ground_truth).sum(axis=1))
    ignorance_to_predictions = np.arccos((ignorance * predictions).sum(axis=1))
    truth_to_predictions = np.arccos((ground_truth * predictions).sum(axis=1))

    def objective(t):
        theta = ignorance_to_truth
        geodesic_cosine = np.clip((
            np.cos(ignorance_to_predictions) * np.sin((1 - t) * theta) / np.sin(theta)
            + np.cos(truth_to_predictions) * np.sin(t * theta) / np.sin(theta)
        ), 0.0, 1.0)
        distance = np.arccos(geodesic_cosine)
        return distance.sum()
    result = optimize.minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")
    return float(np.clip(result.x, 0.0, 1.0))

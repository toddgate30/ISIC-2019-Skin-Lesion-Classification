import torch
from diagnostics import *
import wandb
import numpy as np
from scipy import optimize
from pathlib import Path
from datetime import datetime

class DiagnosticManager():
    def __init__(self, config):
        self.diagnostic_interval = config.get("diagnostic_interval", "log_interval")
        if self.diagnostic_interval == "log_interval":
            pass
        else:
            raise NotImplementedError(f"Diagnostic Interval {self.diagnostic_interval} not implemented.")
        self.next_diagnostic_step = 1
        self.best_val_acc = 0.0
        self.save_parent_dir = Path(config.get("save_dir", "./experiments"))
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_dir = self.save_parent_dir / timestamp
        self.save_dir.mkdir(parents=True, exist_ok=True)


    def _should_run(self, step):
        GROWTH_FACTOR = 1.4
        if step >= self.next_diagnostic_step:
            self.next_diagnostic_step = int(self.next_diagnostic_step * GROWTH_FACTOR)
            return True
        else:
            return False

    def conditional_run(self, step, epoch, model, loss_function, train_metrics_loader, val_loader, device, optimizer):
        if self._should_run(self, step):
            self.run_diagnostics(step, epoch, model, loss_function, train_metrics_loader, val_loader, device, optimizer, final_log=False)

    def forced_run(self, step, epoch, model, loss_function, train_metrics_loader, val_loader, device, optimizer, final_log=False):
        self.run_diagnostics(step, epoch, model, loss_function, train_metrics_loader, val_loader, device, optimizer, final_log=final_log)

    def run_diagnostics(self, step, epoch, model, loss_function, train_metrics_loader, val_loader, device, optimizer, final_log=False):
        train_loss, train_acc, train_progress = self.calculate_diagnotics(model, loss_function, train_metrics_loader, device)
        val_loss, val_acc, val_progress = self.calculate_diagnotics(model, loss_function, val_loader, device)
        self._log_metrics(step, epoch, train_loss, train_acc, train_progress, val_loss, val_acc, val_progress)
        self.save_model(step, epoch, optimizer, model, final_log, val_acc)

    def calculate_diagnotics(self, model, loss_function, dataloader, device):
        loss = 0
        acc = 0
        progress = 0
        model.eval()
        with torch.no_grad():
            batches = 0
            for images, labels in dataloader:
                batches += 1
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss += loss_function(logits, labels)
                predictions = logits.argmax(dim=1)
                acc += (predictions == labels).float().mean()
                log_probs = torch.log_softmax(logits, dim=1)
                progress += self._calculate_progress(log_probs, labels)

            loss /= batches
            acc /= batches
            progress /= batches
        return loss, acc, progress

    def _calculate_progress(self, log_probs, labels):
        probs = torch.exp(log_probs.detach()).cpu()
        probs = probs / probs.sum(Dim=1, keepdim=True)

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
                np.cos(ignorance_to_predictions) * np.sin((1 - t) * theta / np.sin(theta))
                + np.cos(truth_to_predictions) * np.sin(t * theta) / (np.sin(theta))
            ), 0.0, 1.0)
            distance = np.arccos(geodesic_cosine)
            return distance.sum()
        result = optimize.minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")
        return float(np.clip(result.x, 0.0, 1.0))
    
    def _log_metrics(self, step, epoch, train_loss, train_acc, train_progress, val_loss, val_acc, val_progress):
        wandb.log({
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "train/progress": train_progress,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "val/progress": val_progress,
            "epoch": epoch
        }, step=step)

    def _should_save(self, val_acc):
        if val_acc >= self.best_val_acc:
            self.best_val_acc = val_acc
            return True

    def save_model(self, step, epoch, optimizer, model, final_log, val_acc):
        model_save_dir = self.save_dir / "snapshots"
        model_save_dir.mkdir(parents=True, exist_ok=True)

        if final_log == True:
            checkpoint_path = model_save_dir / "last_step_checkpoint.pt"
            checkpoint = {
                "step": step,
                "epoch": epoch,
                "val_acc": val_acc,
                "model_state_dict": model.state_dict(),
                "optimizer_state": optimizer.state_dict()
            }
            torch.save(checkpoint, checkpoint_path)
        elif self._should_save(self, val_acc):
            checkpoint_path = model_save_dir / "best_acc_checkpoint.pt"
            checkpoint = {
                "step": step,
                "epoch": epoch,
                "val_acc": val_acc,
                "model_state_dict": model.state_dict(),
                "optimizer_state": optimizer.state_dict()
            }
            torch.save(checkpoint, checkpoint_path)



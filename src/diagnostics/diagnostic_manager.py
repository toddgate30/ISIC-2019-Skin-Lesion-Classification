import torch
import wandb
import numpy as np
from scipy import optimize
from pathlib import Path
import diagnostics

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
        run_dir = config["run_dir"]
        self.save_dir = self.save_parent_dir / run_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)


    def _should_run(self, step):
        GROWTH_FACTOR = 1.4
        if step >= self.next_diagnostic_step:
            self.next_diagnostic_step = self.next_diagnostic_step + min(500, max(1, int(self.next_diagnostic_step * (GROWTH_FACTOR - 1))))
            return True
        else:
            return False

    def conditional_run(self, step, epoch, context):
        if self._should_run(step):
            self.run_diagnostics(step, epoch, context, final_log=False)

    def forced_run(self, step, epoch, context, final_log=False):
        self.run_diagnostics(step, epoch, context, final_log=final_log)

    def run_diagnostics(self, step, epoch, context, final_log=False):
        train_loss, train_acc, train_balanced_acc, train_progress = self.calculate_diagnotics(context, context.train_metrics_loader)
        val_loss, val_acc, val_balanced_acc, val_progress = self.calculate_diagnotics(context, context.val_loader)
        self._log_metrics(step, epoch, train_loss, train_acc, train_balanced_acc, train_progress, val_loss, val_acc, val_balanced_acc val_progress)
        self.save_model(step, epoch, context, final_log, val_acc)

    def calculate_diagnotics(self, context, dataloader):
        logits, predictions, log_probs, labels = diagnostics.forward_pass(context, dataloader)

        loss = diagnostics.eval_loss(context, logits, labels)
        acc = diagnostics.eval_acc(predictions, labels)
        balanced_acc = diagnostics.eval_balanced_acc(predictions, labels)
        progress = diagnostics.eval_progress(log_probs, labels)
        return loss, acc, balanced_acc, progress
    
    def _log_metrics(self, step, epoch, train_loss, train_acc, train_balanced_acc, train_progress, val_loss, val_acc, val_balanced_acc, val_progress):
        wandb.log({
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "train/balanced accuracy": train_balanced_acc,
            "train/progress": train_progress,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "val/balanced accuracy": val_balanced_acc,
            "val/progress": val_progress,
            "epoch": epoch
        }, step=step)

        print(
            f"Epoch {epoch:3d} | "
            f"Step {step:6d} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2%} | "
            f"Val Bal Acc: {val_balanced_acc:.2%} | "
            f"Val Prog: {val_progress:.2f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2%} | "
            f"Train Bal Acc: {train_balanced_acc:.2%} | "
            f"Train Prog: {train_progress:.2f}"
        )

    def _should_save(self, val_acc):
        if val_acc >= self.best_val_acc:
            self.best_val_acc = val_acc
            return True

    def save_model(self, step, epoch, context, final_log, val_acc):
        model_save_dir = self.save_dir / "snapshots"
        model_save_dir.mkdir(parents=True, exist_ok=True)

        if final_log == True:
            checkpoint_path = model_save_dir / "last_step_checkpoint.pt"
            checkpoint = {
                "step": step,
                "epoch": epoch,
                "val_acc": val_acc,
                "model_state_dict": context.model.state_dict(),
                "optimizer_state": context.optimizer.state_dict()
            }
            torch.save(checkpoint, checkpoint_path)
        elif self._should_save(val_acc):
            checkpoint_path = model_save_dir / "best_acc_checkpoint.pt"
            checkpoint = {
                "step": step,
                "epoch": epoch,
                "val_acc": val_acc,
                "model_state_dict": context.model.state_dict(),
                "optimizer_state": context.optimizer.state_dict()
            }
            torch.save(checkpoint, checkpoint_path)



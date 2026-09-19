import torch
import warnings
from src.diagnostics.save_checkpoint import save_checkpoint

class Trainer():
    """Manages the training loop and lifecycle of a model training run.

    The trainer coordinates model optimization, batch selection, diagnostics,
    checkpointing, and training-state restoration using the components stored
    in the training context.

    Args:
        context: Training context containing the model, optimizer, loss
            function, data loaders, batch selector, and other training
            components.
        diagnostic_manager: Manager responsible for running training
            diagnostics and tracking validation metrics.
        config: Configuration dictionary containing the training settings,
            including the total number of epochs.
    """

    def __init__(self, context, diagnostic_manager, config):
        self.context = context
        self.config = config
        self.total_epochs = config["total_epochs"]
        self.diagnostic_manager = diagnostic_manager
        self.start_epoch = 1
        self.start_step = 0

    def load_state(self, epoch, step, best_val_accuracy):
        """Restore the trainer's state from a saved checkpoint.

        Sets the starting epoch and training step so that training resumes
        after the checkpointed epoch. Also restores the best validation
        accuracy tracked by the diagnostic manager.

        Args:
            epoch: The last completed epoch saved in the checkpoint.
            step: The training step saved in the checkpoint.
            best_val_accuracy: Best validation accuracy achieved before the
                checkpoint was saved.
        """
        self.start_epoch = epoch + 1
        self.start_step = step
        self.diagnostic_manager.best_val_acc = best_val_accuracy

    def before_train(self):
        """Run diagnostics required before training begins.

        Runs a forced diagnostic pass at epoch 0 and training step 0 to
        establish initial metrics before any model updates occur.
        """
        self.diagnostic_manager.forced_run(0, 0, self.context)
    
    def train(self):
        """Train the model for the configured number of epochs.

        For each training step, a meta-batch is loaded and passed to the batch
        selector to determine the examples used for the model update. The
        selected examples are then used for a forward pass, loss computation,
        backpropagation, and optimizer update. Diagnostics are run according
        to the diagnostic manager's configured schedule, and a checkpoint is
        saved after each epoch.

        If training is resumed from a checkpoint, training begins at the
        restored epoch and step rather than starting from the beginning.

        After the final epoch, the learning-rate scheduler is advanced by one
        step when a scheduler is configured.
        """
        print("Starting Training....")
        device = self.context.device
        model = self.context.model
        batch_ratio = self.config.get("batch_ratio", 0.1)
        step = self.start_step

        # Starts epoch loop
        for epoch in range(self.start_epoch, self.total_epochs + 1):
            print(f"\n{'=' * 50}")
            print(f"Training Epoch {epoch}")
            print(f"{'=' * 50}")

            # Starts training loop
            for metabatch_images, metabatch_labels in self.context.train_loader:
                step += 1
                metabatch_images = metabatch_images.to(device)
                metabatch_labels = metabatch_labels.to(device)
                images, labels = self.context.selector(metabatch_images, metabatch_labels, batch_ratio, self.context)

                # Forward Pass
                model.train()
                outputs = model(images)

                # Calculate Loss
                loss = self.context.loss_function(outputs, labels).mean()

                # Backpropagation
                self.context.optimizer.zero_grad()
                loss.backward()
                self.context.optimizer.step()

                # Run metrics
                self.diagnostic_manager.conditional_run(step, epoch, self.context)

            # Saves checpoint at the end of epoch (checks to see if it is a saving epoch or if it is the best model)
            save_checkpoint(self.config, self.context, epoch, step, self.diagnostic_manager.best_val_acc)
        self.final_step_count = step
        if self.context.lr_scheduler is not None:
            self.context.lr_scheduler.step()
    
    def after_train(self):
        """Run final diagnostics after training is complete.

        Forces a final diagnostic pass using the final training step and
        epoch, and marks the diagnostic run as the final log entry.
        """
        self.diagnostic_manager.forced_run(self.final_step_count, self.total_epochs, self.context, final_log=True)
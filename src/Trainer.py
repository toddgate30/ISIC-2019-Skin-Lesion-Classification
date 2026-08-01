import torch
import warnings
from diagnostics.save_checkpoint import save_checkpoint

class Trainer():
    def __init__(self, context, diagnostic_manager, config):
        self.context = context
        self.config = config
        self.total_epochs = config["total_epochs"]
        self.diagnostic_manager = diagnostic_manager
        self.start_epoch = 1
        self.start_step = 0

    def load_state(self, epoch, step, best_val_accuracy):
        self.start_epoch = epoch + 1
        self.start_step = step
        self.diagnostic_manager.best_val_acc = best_val_accuracy

    def before_train(self):
        self.diagnostic_manager.forced_run(0, 0, self.context)
    
    def train(self):
        print("Starting Training....")
        device = self.context.device
        model = self.context.model
        batch_ratio = self.config.get("batch_ratio", 0.1)
        step = self.start_step
        for epoch in range(self.start_epoch, self.total_epochs + 1):
            print(f"\n{'=' * 50}")
            print(f"Training Epoch {epoch}")
            print(f"{'=' * 50}")
            # Training Loop
            model.train()

            for metabatch_images, metabatch_labels in self.context.train_loader:
                step += 1
                metabatch_images = metabatch_images.to(device)
                metabatch_labels = metabatch_labels.to(device)

                images, labels = self.context.selector(metabatch_images, metabatch_labels, batch_ratio)

                # Forward Pass
                outputs = model(images)

                # Calculate Loss
                loss = self.context.loss_function(outputs, labels)

                # Backpropagation
                self.context.optimizer.zero_grad()
                loss.backward()
                self.context.optimizer.step()

                self.diagnostic_manager.conditional_run(step, epoch, self.context)
            save_checkpoint(self.config, self.context, epoch, step, self.diagnostic_manager.best_val_acc)
        self.final_step_count = step
        if self.context.lr_scheduler is not None:
            self.context.lr_scheduler.step()
    
    def after_train(self):
        self.diagnostic_manager.forced_run(self.final_step_count, self.total_epochs, self.context, final_log=True)
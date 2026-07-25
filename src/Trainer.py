import torch
import warnings

class Trainer():
    def __init__(self, context, diagnostic_manager, config):
        self.context = context
        self.config = config
        self.total_epochs = config["total_epochs"]
        self.diagnostic_manager = diagnostic_manager
        

    def before_train(self):
        self.diagnostic_manager.forced_run(0, 0, self.context)
    
    def train(self):
        print("Starting Training....")
        device = self.context.device
        model = self.context.model
        batch_ratio = self.config.get("batch_ratio", 0.1)
        step = 0
        for epoch in range(1, self.total_epochs + 1):
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
        self.final_step_count = step
    
    def after_train(self):
        self.diagnostic_manager.forced_run(self.final_step_count, self.total_epochs, self.context, final_log=True)
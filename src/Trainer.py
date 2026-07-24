import torch
import warnings

class Trainer():
    def __init__(self, model, dataloaders, selector, loss_function, optimizer, diagnostic_manager, config):
        self.model = model
        self.selector = selector
        self.train_loader = dataloaders[0]
        self.train_metrics_loader = dataloaders[1]
        self.val_loader = dataloaders[2]
        self.config = config
        self.total_epochs = config["total_epochs"]
        self.criterion = loss_function
        self.optimizer = optimizer
        self.diagnostic_manager = diagnostic_manager

        if torch.cuda.is_available():
            self.device = torch.deivce("cuda")
        else:
            warnings.warn("No cuda device found. Training on CPU")
            self.device = torch.device("cpu")

    def before_train(self):
        self.diagnostic_manager.forced_run(0, 0, self.model, self.criterion, self.train_metrics_loader, self.val_loader, self.device)
    
    def train(self):
        device = self.device
        model = self.model
        step = 0
        for epoch in range(1, self.total_epochs + 1):
            # Training Loop
            model.train()

            train_loss = 0.0

            for metabatch_images, metabatch_labels in self.train_loader:
                step += 1
                metabatch_images = metabatch_images.to(device)
                metabatch_labels = metabatch_labels.to(device)

                batch_ratio = self.config.get("batch_ratio", 0.1)
                images, labels = self.selector(metabatch_images, metabatch_labels, batch_ratio)

                # Forward Pass
                outputs = model(images)

                # Calculate Loss
                loss = self.criterion(outputs, labels)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.diagnostic_manager.conditional_run(step, epoch, model, self.criterion, self.train_metrics_loader, self.val_loader, device)
        self.final_step_count = step
    
    def after_train(self):
        self.diagnostic_manager.forced_run(self.final_step_count, self.total_epochs, self.model, self.criterion, self.train_metrics_loader, self.val_loader, self.device)
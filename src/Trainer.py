import torch
import warnings

class Trainer():
    def __init__(self, model, dataloaders, selector, loss_function, optimizer, config):
        self.model = model
        self.selector = selector
        self.train_loader = dataloaders[0]
        self.val_loader = dataloaders[1]
        self.config = config
        self.total_epochs = config["total_epochs"]
        self.criterion = loss_function
        self.optimizer = optimizer

        if torch.cuda.is_available():
            self.device = torch.deivce("cuda")
        else:
            warnings.warn("No cuda device found. Training on CPU")
            self.device = torch.device("cpu")
        self.model.to(self.device)

    def before_train():
        raise NotImplementedError
    
    def train(self):
        model = self.model
        device = self.device
        for epoch in range(self.total_epochs):
            # Training Loop
            model.train()

            train_loss = 0.0

            for metabatch_images, metabatch_labels in self.train_loader:
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
    
    def after_train():
        raise NotImplementedError
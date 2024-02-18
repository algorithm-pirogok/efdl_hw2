import torch
from torch import nn
from tqdm.auto import tqdm
import wandb

from unet import Unet

from dataset import get_train_data
from typing import Literal

class StaticScaler:
    def __init__(self):
        self.scale_coeff = 2**8
    
    def scale(self, unscale_loss: torch.Tensor):
        return unscale_loss * self.scale_coeff
    
    def step(self, optimizer: torch.optim.Optimizer):
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad = param.grad / self.scale_coeff
                    if not torch.isfinite(param.grad).all().item():
                        self._update(False)
                        return
        optimizer.step()
        self._update(True)    
    
    def _update(self, mode):
        pass

class DynamicScaler(StaticScaler):
    def __init__(self):
        super().__init__()
        self.cnt = 0
        
    def _update(self, mode):
        if not mode:
            self.scale_coeff //= 2
            self.cnt = 0
        else:
            if self.cnt == 10:
                self.scale_coeff *= 2
            self.cnt = (self.cnt + 1) % 11
            


def train_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mode_of_precision: Literal["base", "static", "dynamic"] = "static"
) -> None:
    model.train()

    if mode_of_precision == "static":
        scaler = StaticScaler()
    elif mode_of_precision == "dynamic":
        scaler = DynamicScaler()
    elif mode_of_precision == "base":
        scaler = None
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)             
            else:
                loss.backward()
                optimizer.step()
        optimizer.zero_grad()

        accuracy = ((outputs > 0.5) == labels).float().mean()
        
        wandb.log({"Loss": loss.item(), 
                   "Accuracy": accuracy.item() * 100, "Scale": scaler.scale_coeff if scaler is not None else 1})
        print({"Loss": loss.item(), 
                   "Accuracy": accuracy.item() * 100, "Scale": scaler.scale_coeff if scaler is not None else 1})
        pbar.set_description(f"Loss: {round(loss.item(), 4)} " f"Accuracy: {round(accuracy.item() * 100, 4)}")
        
    wandb.log({"LossFinal": loss.item(), 
               "AccuracyFinal": accuracy.item() * 100})


def train(mode_of_precision):
    device = torch.device("cuda:0")
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = get_train_data()

    num_epochs = 5
    for epoch in range(0, num_epochs):
        train_epoch(train_loader, model, criterion, optimizer, device=device, mode_of_precision=mode_of_precision)

if __name__ == '__main__':
    wandb.login() # Добавили wandb
    wandb.init(project='hw_2', name="dynamic")

    train(mode_of_precision="dynamic")
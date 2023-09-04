import torch
import torchvision
from dataset import drivedataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
def load_pretrained_model(model):
    print("=> Loading model")
    model = torch.load(model, map_location=torch.device('cpu'))
    return model
    
def get_loaders(train_dir, train_maskdir, val_dir, val_maskdir, batch_size, train_transform, val_transform):
    train_ds = drivedataset(images_dir=train_dir, masks_dir=train_maskdir, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    val_ds = drivedataset(images_dir=val_dir, masks_dir=val_maskdir, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def plot_metrics(train_loss, val_loss, epochs, experiment_name):
    epochs = np.arange(0, epochs)
    
    plt.plot(epochs, train_loss)
    plt.plot(epochs, val_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("Training and Validation Loss Curves")
    
    plt.savefig(f'checkpoints/{experiment_name}/losses.jpg')
    
    print(f'Loss curves stored at checkpoints/{experiment_name}/losses.jpg')
    
    return None

def test(loader, model, device):
    model = model.to(device=device)
    model.eval()
    test_loss = 0
    for i, (input, label) in enumerate(loader):
        pass
    
    return None

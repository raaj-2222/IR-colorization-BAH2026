import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------
# 1. Data Loader for Colorization
# -----------------------------------------
class ColorDataset(Dataset):
    def __init__(self, patches_dir):
        self.sample_dirs = glob.glob(os.path.join(patches_dir, '*', 'sample_*'))
        
    def __len__(self):
        return len(self.sample_dirs)
    
    def __getitem__(self, idx):
        sample_path = self.sample_dirs[idx]
        
        # Input: 100m TIR (1 channel), Target: 100m RGB (3 channels)
        input_path = os.path.join(sample_path, 'tir_100m_512.npy')
        target_path = os.path.join(sample_path, 'rgb_100m_512.npy')
        
        x = np.load(input_path).astype(np.float32)
        y = np.load(target_path).astype(np.float32)
        
        return torch.from_numpy(x), torch.from_numpy(y)

# -----------------------------------------
# 2. U-Net Generator (TIR -> RGB)
# -----------------------------------------
class ColorNet(nn.Module):
    def __init__(self):
        super(ColorNet, self).__init__()
        
        # Encoder (Downsample)
        self.enc1 = nn.Sequential(nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        
        # Bottleneck
        self.bot = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU())
        
        # Decoder (Upsample back to original size)
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU())
        # Final output layer needs 3 channels for RGB
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bot(e2)
        d1 = self.dec1(b)
        # Skip connection: add encoder features to decoder
        d1 = d1 + e1 
        out = self.dec2(d1)
        return out

# -----------------------------------------
# 3. Training Loop
# -----------------------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Colorization on: {device}")
    
    dataset = ColorDataset(patches_dir='output/patches')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    if len(dataset) == 0:
        print("No samples found!")
        return

    model = ColorNet().to(device)
    # L1 Loss (Mean Absolute Error) is standard for image-to-image colorization
    criterion = nn.L1Loss() 
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    epochs = 15
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss/len(dataloader):.4f}")
        
    os.makedirs('output/model_weights', exist_ok=True)
    torch.save(model.state_dict(), 'output/model_weights/color_model.pth')
    print("Colorization weights saved to output/model_weights/color_model.pth")

if __name__ == '__main__':
    train()
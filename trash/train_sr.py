import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------
# 1. Data Loader for your specific patches
# -----------------------------------------
class TIRDataset(Dataset):
    def __init__(self, patches_dir):
        # Find all sample folders (e.g., sample_LC09_mumbai_002)
        self.sample_dirs = glob.glob(os.path.join(patches_dir, '*', 'sample_*'))
        
    def __len__(self):
        return len(self.sample_dirs)
    
    def __getitem__(self, idx):
        sample_path = self.sample_dirs[idx]
        
        # Load the 200m input (1, 256, 256) and 100m target (1, 512, 512)
        # Cast to float32 for PyTorch compatibility
        input_path = os.path.join(sample_path, 'tir_200m.npy')
        target_path = os.path.join(sample_path, 'tir_100m_512.npy')
        
        x = np.load(input_path).astype(np.float32)
        y = np.load(target_path).astype(np.float32)
        
        # Convert to PyTorch tensors
        return torch.from_numpy(x), torch.from_numpy(y)

# -----------------------------------------
# 2. Minimal Super-Resolution Model (2x Upscale)
# -----------------------------------------
class SimpleSRNet(nn.Module):
    def __init__(self):
        super(SimpleSRNet, self).__init__()
        
        # Extract features without changing dimensions
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Upscale by 2x (256 -> 512)
        self.upsample = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.relu_up = nn.ReLU(inplace=True)
        
        # Compress back to 1 grayscale channel
        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.features(x)
        x = self.upsample(x)
        x = self.relu_up(x)
        x = self.final_conv(x)
        return x

# -----------------------------------------
# 3. Training Loop
# -----------------------------------------
def train():
    # Setup device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Load Data
    dataset = TIRDataset(patches_dir='output/patches')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    if len(dataset) == 0:
        print("No samples found! Check your patches directory.")
        return

    # Initialize Model, Loss (MSE for pixel-perfect reconstruction), and Optimizer
    model = SimpleSRNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 10
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss/len(dataloader):.4f}")
        
    # Save the required weights file
    os.makedirs('output/model_weights', exist_ok=True)
    torch.save(model.state_dict(), 'output/model_weights/sr_model.pth')
    print("Training complete. Weights saved to output/model_weights/sr_model.pth")

if __name__ == '__main__':
    train()
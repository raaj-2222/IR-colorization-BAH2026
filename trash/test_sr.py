import os
import glob
import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image

# 1. Redefine the exact same architecture so PyTorch can load the weights
class SimpleSRNet(nn.Module):
    def __init__(self):
        super(SimpleSRNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.relu_up = nn.ReLU(inplace=True)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.features(x)
        x = self.upsample(x)
        x = self.relu_up(x)
        x = self.final_conv(x)
        return x

def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model and Weights
    model = SimpleSRNet().to(device)
    model.load_state_dict(torch.load('output/model_weights/sr_model.pth', map_location=device))
    model.eval() # Set to evaluation mode
    
    # Grab the first available sample from your dataset
    sample_dirs = glob.glob(os.path.join('output', 'patches', '*', 'sample_*'))
    if not sample_dirs:
        print("No samples found. Check output/patches/ directory.")
        return
        
    sample_path = sample_dirs[0]
    input_path = os.path.join(sample_path, 'tir_200m.npy')
    
    # Load the .npy file and shape it for PyTorch (Batch, Channel, Height, Width)
    x_np = np.load(input_path).astype(np.float32)
    x_tensor = torch.from_numpy(x_np).unsqueeze(0).to(device) 
    
    # Run the model
    with torch.no_grad():
        output = model(x_tensor)
        
    # Create an output directory
    os.makedirs('output/test_results', exist_ok=True)
    
    # Normalize the arrays so they save properly as viewable PNGs
    def normalize_for_saving(tensor):
        img = tensor.squeeze(0) # Remove batch dimension
        return (img - img.min()) / (img.max() - img.min() + 1e-8)

    in_img_norm = normalize_for_saving(x_tensor)
    out_img_norm = normalize_for_saving(output)
    
    # Save the input and the prediction
    save_image(in_img_norm, 'output/test_results/input_200m.png')
    save_image(out_img_norm, 'output/test_results/prediction_100m.png')
    
    print("Inference complete! Check output/test_results/ to compare the images.")

if __name__ == '__main__':
    run_inference()
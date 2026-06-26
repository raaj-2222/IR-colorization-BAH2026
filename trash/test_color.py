import os
import glob
import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image

# 1. Redefine the exact U-Net architecture
class ColorNet(nn.Module):
    def __init__(self):
        super(ColorNet, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.bot = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1)) # 3 channels for RGB!

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bot(e2)
        d1 = self.dec1(b)
        out = self.dec2(d1 + e1)
        return out

def run_color_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model and Weights
    model = ColorNet().to(device)
    model.load_state_dict(torch.load('output/model_weights/color_model.pth', map_location=device))
    model.eval() 
    
    # Grab the first available sample
    sample_dirs = glob.glob(os.path.join('output', 'patches', '*', 'sample_*'))
    if not sample_dirs:
        print("No samples found. Check output/patches/ directory.")
        return
        
    sample_path = sample_dirs[0]
    
    # We feed it the 100m TIR image, it predicts the 100m RGB image
    input_path = os.path.join(sample_path, 'tir_100m_512.npy')
    
    x_np = np.load(input_path).astype(np.float32)
    x_tensor = torch.from_numpy(x_np).unsqueeze(0).to(device) 
    
    with torch.no_grad():
        output = model(x_tensor)
        
    os.makedirs('output/test_results', exist_ok=True)
    
    # Normalize for saving as PNG
    def normalize_for_saving(tensor):
        img = tensor.squeeze(0) 
        return (img - img.min()) / (img.max() - img.min() + 1e-8)

    in_img_norm = normalize_for_saving(x_tensor)
    out_img_norm = normalize_for_saving(output)
    
    # Save the input and the prediction
    save_image(in_img_norm, 'output/test_results/color_input_tir.png')
    save_image(out_img_norm, 'output/test_results/color_prediction_rgb.png')
    
    print("Color inference complete! Check output/test_results/ to see the RGB prediction.")

if __name__ == '__main__':
    run_color_inference()
import os
import glob
import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image

# -----------------------------------------
# 1. Architectures (Stages A and B)
# -----------------------------------------
class SimpleSRNet(nn.Module):
    def __init__(self):
        super(SimpleSRNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )
        self.upsample = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.relu_up = nn.ReLU(inplace=True)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        return self.final_conv(self.relu_up(self.upsample(self.features(x))))

class ColorNet(nn.Module):
    def __init__(self):
        super(ColorNet, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.bot = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1))

    def forward(self, x):
        e1 = self.enc1(x)
        return self.dec2(self.dec1(self.bot(self.enc2(e1))) + e1)

# -----------------------------------------
# 2. Execution Pipeline
# -----------------------------------------
def run_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading models...")
    
    # Load SR Model
    sr_model = SimpleSRNet().to(device)
    sr_model.load_state_dict(torch.load('output/model_weights/sr_model.pth', map_location=device))
    sr_model.eval()
    
    # Load Colorization Model
    color_model = ColorNet().to(device)
    color_model.load_state_dict(torch.load('output/model_weights/color_model.pth', map_location=device))
    color_model.eval()

    # Get a 200m Input Patch
    sample_dirs = glob.glob(os.path.join('output', 'patches', '*', 'sample_*'))
    if not sample_dirs:
        print("No samples found!")
        return
        
    input_path = os.path.join(sample_dirs[0], 'tir_200m.npy')
    raw_tir_np = np.load(input_path).astype(np.float32)
    raw_tir_tensor = torch.from_numpy(raw_tir_np).unsqueeze(0).to(device)

    print("Running End-to-End Inference...")
    with torch.no_grad():
        # STAGE A: 200m TIR -> 100m TIR
        super_resolved_tir = sr_model(raw_tir_tensor)
        
        # STAGE B: 100m TIR -> 100m RGB
        colorized_rgb = color_model(super_resolved_tir)

    # -----------------------------------------
    # 3. Save the Sequence
    # -----------------------------------------
    os.makedirs('output/final_pipeline', exist_ok=True)
    
    def norm(t):
        img = t.squeeze(0)
        return (img - img.min()) / (img.max() - img.min() + 1e-8)

    save_image(norm(raw_tir_tensor), 'output/final_pipeline/1_raw_200m.png')
    save_image(norm(super_resolved_tir), 'output/final_pipeline/2_sharpened_100m.png')
    save_image(norm(colorized_rgb), 'output/final_pipeline/3_colorized_100m.png')
    
    print("Pipeline complete! Check output/final_pipeline/ for the 3-step sequence.")

if __name__ == '__main__':
    run_pipeline()
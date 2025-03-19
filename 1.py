import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.transforms.functional import gaussian_blur

# Define StereoHRNet model architecture with reinforced fusion
class StereoHRNet(nn.Module):
    def __init__(self):
        super(StereoHRNet, self).__init__()
        
        self.vis_features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((32, 32))
        )
        
        self.ir_features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((32, 32))
        )

        self.disparity = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def compute_disparity_matrix(self, vis_feats, ir_feats):
        b, c, h, w = vis_feats.shape
        vis_flat = vis_feats.view(b, c, -1)  
        ir_flat = ir_feats.view(b, c, -1)  
        
        disparity_matrix = torch.matmul(vis_flat.transpose(1,2), ir_flat)  
        disparity_matrix = F.softmax(disparity_matrix, dim=-1)
        
        return disparity_matrix.view(b, h, w, h*w)

    def forward(self, vis, ir):
        vis_feats = self.vis_features(vis)  
        ir_feats = self.ir_features(ir)  
        
        disparity_matrix = self.compute_disparity_matrix(vis_feats, ir_feats)
        
        fused_feats = torch.cat([vis_feats, ir_feats], dim=1)
        disparity_map = self.disparity(fused_feats)
        
        return disparity_map, disparity_matrix

class DepthEstimator:
    def __init__(self):
        self.model = StereoHRNet()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

    def estimate_depth(self, vis_img, ir_img):
        with torch.no_grad():
            disparity_map, disparity_matrix = self.model(vis_img, ir_img)
            return disparity_map, disparity_matrix

def process_all_images():
    ir_path = "D:/DL/M3FD_Fusion/ir"
    vis_path = "D:/DL/M3FD_Fusion/vis"
    
    depth_est = DepthEstimator()
    
    ir_files = sorted(os.listdir(ir_path))[:3]
    vis_files = sorted(os.listdir(vis_path))[:3]
    
    for ir_file, vis_file in zip(ir_files, vis_files):
        ir_img = cv2.imread(os.path.join(ir_path, ir_file), cv2.IMREAD_GRAYSCALE)
        vis_img = cv2.imread(os.path.join(vis_path, vis_file))
        
        ir_tensor = torch.from_numpy(ir_img).float().unsqueeze(0).unsqueeze(0) / 255.0
        vis_tensor = torch.from_numpy(vis_img).float().permute(2,0,1).unsqueeze(0) / 255.0
        
        if torch.cuda.is_available():
            ir_tensor = ir_tensor.cuda()
            vis_tensor = vis_tensor.cuda()
            
        disparity_map, disparity_matrix = depth_est.estimate_depth(vis_tensor, ir_tensor)
        
        disparity_map = disparity_map.cpu().numpy()[0,0]
        disparity_map = (disparity_map * 255).astype(np.uint8)
        
        depth_contours = cv2.Laplacian(disparity_map, cv2.CV_8U)
        
        plt.figure(figsize=(10,5))
        plt.subplot(131)
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title('Visible Image')
        plt.axis('off')
        
        plt.subplot(132) 
        plt.imshow(ir_img, cmap='gray')
        plt.title('IR Image')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(disparity_map, cmap='jet')
        plt.title('Disparity Map')
        plt.axis('off')
        
        plt.show()

if __name__ == "__main__":
    process_all_images()

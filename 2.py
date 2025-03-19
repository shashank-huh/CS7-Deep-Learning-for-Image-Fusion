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
    def __init__(self, config):
        """
        Stereo matching network using HRNet backbone with reinforced fusion
        """
        super(StereoHRNet, self).__init__()
        
        # Feature extraction for visible and IR branches
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

        # Segmentation branch
        self.segmentation = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=1),  # 2 classes: foreground/background
        )
        
        # Reinforced fusion module
        self.fusion_attention = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1), # Changed output channels to match input
            nn.Softmax(dim=1)
        )
        
        # Classifiers for correlation and concatenation branches
        self.correlation_cls = nn.Sequential(
            nn.Linear(65536, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        
        self.concat_cls = nn.Sequential(
            nn.Linear(131072, 1024),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

        # Disparity estimation branch
        self.disparity = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def compute_disparity_matrix(self, vis_feats, ir_feats):
        """
        Compute disparity matrix between visible and IR features
        """
        b, c, h, w = vis_feats.shape
        vis_flat = vis_feats.view(b, c, -1)  # [B,C,H*W]
        ir_flat = ir_feats.view(b, c, -1)  # [B,C,H*W]
        
        # Compute correlation matrix
        disparity_matrix = torch.matmul(vis_flat.transpose(1,2), ir_flat)  # [B,H*W,H*W]
        disparity_matrix = F.softmax(disparity_matrix, dim=-1)
        
        return disparity_matrix.view(b, h*w, h, w)

    def forward(self, vis, ir):
        """
        Forward pass with reinforced fusion and disparity estimation
        Args:
            vis: Visible image tensor [B,3,H,W]
            ir: IR image tensor [B,1,H,W] 
        Returns:
            correlation, concatenation features, segmentation mask, disparity map
        """
        # Extract features
        vis_feats = self.vis_features(vis)  # [B,64,32,32]
        ir_feats = self.ir_features(ir)  # [B,64,32,32]
        
        # Compute disparity matrix
        disparity_matrix = self.compute_disparity_matrix(vis_feats, ir_feats)
        
        # Fuse features with attention
        fused_feats = torch.cat([vis_feats, ir_feats], dim=1)  # [B,128,32,32]
        attention_weights = self.fusion_attention(fused_feats)  # Now outputs [B,128,32,32]
        fused_feats = fused_feats * attention_weights
        
        # Generate segmentation mask
        seg_mask = self.segmentation(fused_feats)
        
        # Correlation branch with reinforced features
        vis_flat = vis_feats.view(vis_feats.size(0), -1)  # [B,65536]
        ir_flat = ir_feats.view(ir_feats.size(0), -1)  # [B,65536]
        correlation = torch.matmul(vis_flat.unsqueeze(1), ir_flat.unsqueeze(2))
        correlation = correlation.squeeze()
        correlation = self.correlation_cls(vis_flat)  # [B,128]
        
        # Concatenation branch with reinforced features
        concat_feats = fused_feats.view(fused_feats.size(0), -1)  # [B,131072]
        concatenation = self.concat_cls(concat_feats)  # [B,128]
        
        # Estimate disparity map
        disparity_map = self.disparity(fused_feats)
        
        return correlation, concatenation, correlation, concatenation, seg_mask, disparity_map, disparity_matrix

class DepthEstimator:
    def __init__(self):
        self.model = StereoHRNet(None)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

    def estimate_depth(self, vis_img, ir_img):
        """
        Estimate depth with reinforced fusion
        Args:
            vis_img: Visible image tensor [B,3,H,W]
            ir_img: IR image tensor [B,1,H,W]
        Returns:
            Depth map, thermal aura, segmentation mask, disparity matrix
        """
        with torch.no_grad():
            # Get features and predictions
            corr, concat, corr_s1, concat_s1, seg_mask, disparity_map, disparity_matrix = self.model(vis_img, ir_img)
            
            # Combine features with reinforcement
            fused_features = torch.cat([corr, concat], dim=1)  # [B,256]
            
            h, w = vis_img.shape[2:]
            
            # Generate enhanced depth map
            depth_map = fused_features.view(-1, 16, 16, 1)
            depth_map = depth_map.permute(0, 3, 1, 2)  # [B,1,16,16]
            depth_map = F.interpolate(depth_map, size=(h,w), mode='bilinear', align_corners=True)
            
            # Interpolate segmentation mask to match depth map size
            seg_mask = F.interpolate(seg_mask, size=(h,w), mode='bilinear', align_corners=True)
            
            # Create thermal aura with segmentation guidance
            thermal_aura = gaussian_blur(depth_map * F.softmax(seg_mask, dim=1)[:,1:], kernel_size=15, sigma=5.0)
            
            # Normalize predictions
            depth_map = F.normalize(depth_map, p=2, dim=1)
            thermal_aura = F.normalize(thermal_aura, p=2, dim=1)
            
            return depth_map, thermal_aura, seg_mask, disparity_matrix

def process_all_images():
    """
    Process image pairs with reinforced fusion and visualization
    """
    # Paths
    ir_path = r"D:\DL\M3FD_Fusion\ir"
    vis_path = r"D:\DL\M3FD_Fusion\vis"
    
    depth_est = DepthEstimator()
    
    ir_files = sorted(os.listdir(ir_path))[:3]
    vis_files = sorted(os.listdir(vis_path))[:3]
    
    for ir_file, vis_file in zip(ir_files, vis_files):
        # Load and preprocess
        ir_img = cv2.imread(os.path.join(ir_path, ir_file), cv2.IMREAD_GRAYSCALE)
        vis_img = cv2.imread(os.path.join(vis_path, vis_file))
        
        ir_tensor = torch.from_numpy(ir_img).float().unsqueeze(0).unsqueeze(0) / 255.0
        vis_tensor = torch.from_numpy(vis_img).float().permute(2,0,1).unsqueeze(0) / 255.0
        
        if torch.cuda.is_available():
            ir_tensor = ir_tensor.cuda()
            vis_tensor = vis_tensor.cuda()
            
        # Get predictions
        depth_map, thermal_aura, seg_mask, disparity_matrix = depth_est.estimate_depth(vis_tensor, ir_tensor)
        
        # Convert to numpy
        depth_map = depth_map.cpu().numpy()[0,0]
        thermal_aura = thermal_aura.cpu().numpy()[0,0]
        seg_mask = F.softmax(seg_mask, dim=1).cpu().numpy()[0,1]
        disparity_matrix = disparity_matrix.cpu().numpy()[0,0]
        
        # Visualizations
        thermal_viz = cv2.applyColorMap((thermal_aura * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        
        alpha = 0.7
        ir_thermal = cv2.addWeighted(cv2.cvtColor(ir_img, cv2.COLOR_GRAY2BGR), 1-alpha, thermal_viz, alpha, 0)
        
        depth_contours = cv2.Laplacian((depth_map * 255).astype(np.uint8), cv2.CV_8U)
        ir_thermal_depth = cv2.addWeighted(ir_thermal, 0.8,
                                         cv2.cvtColor(depth_contours, cv2.COLOR_GRAY2BGR), 0.2, 0)
        
        # Plot results
        plt.figure(figsize=(20,10))
        
        plt.subplot(231)
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title('Visible Image')
        plt.axis('off')
        
        plt.subplot(232) 
        plt.imshow(ir_img, cmap='gray')
        plt.title('IR Image')
        plt.axis('off')
        
        '''plt.subplot(233)
        plt.imshow(cv2.cvtColor(ir_thermal, cv2.COLOR_BGR2RGB))
        plt.title('Thermal Aura')
        plt.axis('off')
        
        plt.subplot(234)
        plt.imshow(cv2.cvtColor(ir_thermal_depth, cv2.COLOR_BGR2RGB))
        plt.title('Depth-Enhanced Thermal')
        plt.axis('off')'''
        
        plt.subplot(233)
        plt.imshow(seg_mask, cmap='jet')
        plt.title('Segmentation Mask')
        plt.axis('off')
        
        plt.subplot(234)
        plt.imshow(disparity_matrix, cmap='jet')
        plt.title('Disparity Matrix')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    process_all_images()
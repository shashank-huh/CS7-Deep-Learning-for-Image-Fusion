import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.transforms.functional import gaussian_blur

class SimpleHRNet(nn.Module):
    """Simplified HRNet backbone for feature extraction"""
    def __init__(self):
        super(SimpleHRNet, self).__init__()
        # Feature extraction backbone
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
    def forward(self, x):
        # Adapt input channels if needed (for IR images)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            
        features = []
        
        x1 = F.relu(self.bn1(self.conv1(x)))
        features.append(x1)
        
        x2 = F.relu(self.bn2(self.conv2(x1)))
        features.append(x2)
        
        x3 = F.relu(self.bn3(self.conv3(x2)))
        features.append(x3)
        
        x4 = F.relu(self.bn4(self.conv4(x3)))
        features.append(x4)
        
        return features

class ReinforcedFusionModule(nn.Module):
    """Module for reinforced fusion of visible and IR features"""
    def __init__(self, in_channels):
        super(ReinforcedFusionModule, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, vis_feat, ir_feat):
        combined = torch.cat([vis_feat, ir_feat], dim=1)
        attention_weights = self.attention(combined)
        
        # Split attention weights for visible and IR branches
        vis_weights, ir_weights = torch.split(attention_weights, vis_feat.size(1), dim=1)
        
        # Apply attention weights
        vis_enhanced = vis_feat * vis_weights
        ir_enhanced = ir_feat * ir_weights
        
        # Reinforced fusion
        fused = vis_enhanced + ir_enhanced
        
        return fused, vis_enhanced, ir_enhanced

class SegmentationModule(nn.Module):
    """Module for segmentation in the Mutually Reinforcing approach"""
    def __init__(self, in_channels):
        super(SegmentationModule, self).__init__()
        
        self.seg_layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(),
            nn.Conv2d(in_channels//2, in_channels//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(),
            nn.Conv2d(in_channels//4, 2, kernel_size=1)  # 2 classes: foreground/background
        )
        
    def forward(self, x):
        return self.seg_layers(x)

class DisparityModule(nn.Module):
    """Module for disparity estimation with reinforced features"""
    def __init__(self, in_channels):
        super(DisparityModule, self).__init__()
        
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(),
            nn.Conv2d(in_channels//2, in_channels//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU()
        )
        
    def forward(self, vis_feat, ir_feat, seg_mask=None):
        # Refine features
        vis_refined = self.refine(vis_feat)
        ir_refined = self.refine(ir_feat)
        
        # Apply segmentation guidance if available
        if seg_mask is not None:
            # Resize segmentation mask to match feature size
            seg_mask_resized = F.interpolate(seg_mask, size=(vis_refined.size(2), vis_refined.size(3)), 
                                           mode='bilinear', align_corners=True)
            
            # Enhance features with segmentation mask
            vis_refined = vis_refined * (1.0 + 0.3 * seg_mask_resized)
            ir_refined = ir_refined * (1.0 + 0.3 * seg_mask_resized)
        
        # Compute disparity matrix
        b, c, h, w = vis_refined.size()
        vis_flat = vis_refined.view(b, c, -1)  # [B, C, H*W]
        ir_flat = ir_refined.view(b, c, -1)    # [B, C, H*W]
        
        # Compute similarity matrix with temperature scaling
        temperature = 20.0  # Increased for sharper contrast
        similarity = torch.matmul(vis_flat.transpose(1, 2), ir_flat) * temperature
        
        # Apply softmax for better contrast
        disparity_matrix = F.softmax(similarity, dim=2)
        
        # Reshape to [B, H*W, H, W]
        disparity_matrix = disparity_matrix.view(b, h*w, h, w)
        
        return disparity_matrix, vis_refined, ir_refined

class DepthModule(nn.Module):
    """Module for depth estimation with fused features"""
    def __init__(self, in_channels):
        super(DepthModule, self).__init__()
        
        self.depth_layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(),
            nn.Conv2d(in_channels//2, in_channels//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(),
            nn.Conv2d(in_channels//4, 1, kernel_size=1)
        )
        
    def forward(self, x):
        return self.depth_layers(x)

class StereoHRNet(nn.Module):
    """Complete network for stereo matching with reinforced fusion"""
    def __init__(self):
        super(StereoHRNet, self).__init__()
        
        # Feature extraction backbone
        self.backbone = SimpleHRNet()
        
        # Reinforced fusion module
        self.fusion = ReinforcedFusionModule(512)
        
        # Segmentation module
        self.segmentation = SegmentationModule(512*2)
        
        # Disparity estimation module
        self.disparity = DisparityModule(512)
        
        # Depth estimation module
        self.depth = DepthModule(512)
        
    def forward(self, vis_img, ir_img):
        # Extract features
        vis_feats = self.backbone(vis_img)
        ir_feats = self.backbone(ir_img)
        
        # Use deepest features
        vis_feat = vis_feats[-1]
        ir_feat = ir_feats[-1]
        
        # Compute segmentation mask for mutual reinforcement
        combined_feats = torch.cat([vis_feat, ir_feat], dim=1)
        seg_logits = self.segmentation(combined_feats)
        seg_mask = F.softmax(seg_logits, dim=1)[:, 1:2]  # Get foreground probability
        
        # Compute disparity matrix with segmentation guidance
        disparity_matrix, vis_refined, ir_refined = self.disparity(vis_feat, ir_feat, seg_mask)
        
        # Apply reinforced fusion
        fused_feats, vis_enhanced, ir_enhanced = self.fusion(vis_feat, ir_feat)
        
        # Enhance fusion with segmentation mask
        fused_feats = fused_feats * (1.0 + 0.3 * F.interpolate(seg_mask, 
                                                               size=(fused_feats.size(2), fused_feats.size(3)), 
                                                               mode='bilinear', align_corners=True))
        
        # Generate depth map
        depth_map = self.depth(fused_feats)
        
        return depth_map, disparity_matrix, seg_logits, fused_feats

class DepthEstimator:
    """Main class for depth estimation using the StereoHRNet model"""
    def __init__(self):
        self.model = StereoHRNet()
        
        # Check if pretrained HRNet weights exist
        pretrained_path = r"D:\DL\DepthEst\hrnetv2_w48_imagenet_pretrained.pth"
        if os.path.exists(pretrained_path):
            print(f"Found pretrained HRNet weights at {pretrained_path}")
            try:
                pretrained_weights = torch.load(pretrained_path)
                # Filter and adapt weights for backbone
                backbone_dict = self.model.backbone.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_weights.items() 
                                  if k in backbone_dict and v.shape == backbone_dict[k].shape}
                
                if pretrained_dict:
                    print(f"Loaded {len(pretrained_dict)}/{len(backbone_dict)} layers from pretrained model")
                    backbone_dict.update(pretrained_dict)
                    self.model.backbone.load_state_dict(backbone_dict)
                else:
                    print("Could not load any weights from pretrained model (architecture mismatch)")
            except Exception as e:
                print(f"Error loading pretrained weights: {e}")
                print("Using randomly initialized model instead")
        else:
            print("Pretrained HRNet weights not found, using randomly initialized model")
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        self.model.eval()
    
    def apply_guided_filter(self, x, guide, radius=1, eps=1e-4):
        """Apply a simplified guided filter for edge-preserving smoothing"""
        # Simple approximation of guided filtering using average pooling
        mean_x = F.avg_pool2d(x, kernel_size=2*radius+1, stride=1, padding=radius)
        mean_guide = F.avg_pool2d(guide, kernel_size=2*radius+1, stride=1, padding=radius)
        
        # Apply filtering
        return mean_x + (x - mean_x) * (guide - mean_guide) / (mean_guide + eps)
    
    def estimate_depth(self, vis_img, ir_img):
        """
        Estimate depth and disparity using the StereoHRNet model
        Args:
            vis_img: Visible image tensor [B,3,H,W]
            ir_img: IR image tensor [B,1,H,W]
        Returns:
            depth_map, disparity_matrix, seg_mask, thermal_aura
        """
        with torch.no_grad():
            # Get predictions from model
            depth_map, disparity_matrix, seg_logits, fused_feats = self.model(vis_img, ir_img)
            
            # Get segmentation mask
            seg_mask = F.softmax(seg_logits, dim=1)[:, 1:2]
            
            # Resize outputs to match input size
            h, w = vis_img.shape[2:]
            depth_map = F.interpolate(depth_map, size=(h, w), mode='bilinear', align_corners=True)
            seg_mask = F.interpolate(seg_mask, size=(h, w), mode='bilinear', align_corners=True)
            
            # Apply edge-preserving smoothing to depth map
            depth_map = self.apply_guided_filter(depth_map, vis_img)
            
            # Apply sigmoid to constrain depth map to [0, 1]
            depth_map = torch.sigmoid(depth_map)
            
            # Create thermal aura with segmentation guidance
            thermal_aura = gaussian_blur(depth_map * seg_mask, kernel_size=15, sigma=5.0)
            
            # Normalize disparity matrix for visualization
            # Extract a single channel from the disparity matrix for visualization
            disparity_vis = torch.mean(disparity_matrix, dim=1, keepdim=True)
            disparity_vis = F.interpolate(disparity_vis, size=(h, w), mode='bilinear', align_corners=True)
            
            # Ensure disparity is normalized
            disparity_vis = (disparity_vis - disparity_vis.min()) / (disparity_vis.max() - disparity_vis.min() + 1e-6)
            
            return depth_map, disparity_vis, seg_mask, thermal_aura, disparity_matrix

def process_all_images():
    """Process image pairs and visualize results"""
    # Paths
    ir_path = r"D:\DL\M3FD_Fusion\ir"
    vis_path = r"D:\DL\M3FD_Fusion\vis"
    
    # Create output directory
    output_dir = r"D:\DL\M3FD_Fusion\output"
    os.makedirs(output_dir, exist_ok=True)
    
    depth_est = DepthEstimator()
    
    # Check if paths exist
    if not os.path.exists(ir_path) or not os.path.exists(vis_path):
        print("Input directories not found. Using sample data for demonstration...")
        # Create sample data
        h, w = 256, 256
        
        # Create sample visible image
        vis_img = np.zeros((h, w, 3), dtype=np.uint8)
        # Add some shapes
        cv2.rectangle(vis_img, (50, 50), (150, 150), (0, 0, 255), -1)  # Red rectangle
        cv2.circle(vis_img, (200, 100), 50, (0, 255, 0), -1)  # Green circle
        cv2.line(vis_img, (0, 200), (255, 200), (255, 0, 0), 5)  # Blue line
        
        # Create sample IR image
        ir_img = np.zeros((h, w), dtype=np.uint8)
        # Add some shapes with different brightness
        cv2.rectangle(ir_img, (60, 60), (140, 140), 255, -1)  # White rectangle
        cv2.circle(ir_img, (200, 100), 40, 200, -1)  # Gray circle
        cv2.line(ir_img, (0, 210), (255, 210), 150, 10)  # Gray line
        
        # Create tensors from sample images
        vis_tensor = torch.from_numpy(vis_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        ir_tensor = torch.from_numpy(ir_img).float().unsqueeze(0).unsqueeze(0) / 255.0
        
        if torch.cuda.is_available():
            vis_tensor = vis_tensor.cuda()
            ir_tensor = ir_tensor.cuda()
            
        # Get predictions
        depth_map, disparity, seg_mask, thermal_aura, disparity_matrix = depth_est.estimate_depth(vis_tensor, ir_tensor)
        
        # Convert to numpy
        depth_map = depth_map.cpu().numpy()[0, 0]
        disparity = disparity.cpu().numpy()[0, 0]
        seg_mask = seg_mask.cpu().numpy()[0, 0]
        thermal_aura = thermal_aura.cpu().numpy()[0, 0]
        
        # Apply histogram equalization for better visualization
        depth_map_uint8 = (depth_map * 255).astype(np.uint8)
        disparity_uint8 = (disparity * 255).astype(np.uint8)
        
        depth_map_enhanced = cv2.equalizeHist(depth_map_uint8)
        disparity_enhanced = cv2.equalizeHist(disparity_uint8)
        
        # Normalize back to [0, 1]
        depth_map_enhanced = depth_map_enhanced / 255.0
        disparity_enhanced = disparity_enhanced / 255.0
        
        # Create mutually reinforced fusion
        vis_gray = cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY)
        fused_img = (vis_gray.astype(float) * (1-seg_mask) + ir_img.astype(float) * seg_mask)
        fused_img = fused_img.astype(np.uint8)
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        plt.subplot(231)
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title('Visible Image (Sample)')
        plt.axis('off')
        
        plt.subplot(232)
        plt.imshow(ir_img, cmap='gray')
        plt.title('IR Image (Sample)')
        plt.axis('off')
        
        plt.subplot(233)
        plt.imshow(seg_mask, cmap='hot')
        plt.title('Segmentation Mask')
        plt.axis('off')
        
        plt.subplot(234)
        plt.imshow(depth_map_enhanced, cmap='jet')
        plt.title('Depth Map')
        plt.axis('off')
        
        plt.subplot(235)
        plt.imshow(disparity_enhanced, cmap='viridis')
        plt.title('Disparity Map')
        plt.axis('off')
        
        plt.subplot(236)
        plt.imshow(thermal_aura, cmap='inferno')
        plt.title('Thermal Aura')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sample_results.png"))
        plt.show()
        
        # Save the disparity matrix visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(disparity_enhanced, cmap='plasma')
        plt.colorbar(label='Disparity Value')
        plt.title('Disparity Matrix Visualization')
        plt.savefig(os.path.join(output_dir, "disparity_matrix.png"))
        plt.show()
        
        return
    
    # Process real images if they exist
    ir_files = sorted(os.listdir(ir_path))[:3]  # Process first 3 images
    vis_files = sorted(os.listdir(vis_path))[:3]
    
    if not ir_files or not vis_files:
        print("No image files found in the input directories.")
        return
    
    for i, (ir_file, vis_file) in enumerate(zip(ir_files, vis_files)):
        print(f"Processing image pair {i+1}: {vis_file} and {ir_file}")
        
        # Load images
        ir_img = cv2.imread(os.path.join(ir_path, ir_file), cv2.IMREAD_GRAYSCALE)
        vis_img = cv2.imread(os.path.join(vis_path, vis_file))
        
        if ir_img is None or vis_img is None:
            print(f"Skipping {ir_file}/{vis_file} - could not read images")
            continue
        
        # Resize to same dimensions if needed
        if ir_img.shape[:2] != vis_img.shape[:2]:
            ir_img = cv2.resize(ir_img, (vis_img.shape[1], vis_img.shape[0]))
        
        # Enhance contrast for better results
        ir_img = cv2.equalizeHist(ir_img)
        
        # Convert to tensors
        ir_tensor = torch.from_numpy(ir_img).float().unsqueeze(0).unsqueeze(0) / 255.0
        vis_tensor = torch.from_numpy(vis_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        if torch.cuda.is_available():
            ir_tensor = ir_tensor.cuda()
            vis_tensor = vis_tensor.cuda()
        
        # Get predictions
        depth_map, disparity, seg_mask, thermal_aura, disparity_matrix = depth_est.estimate_depth(vis_tensor, ir_tensor)
        
        # Convert to numpy
        depth_map = depth_map.cpu().numpy()[0, 0]
        disparity = disparity.cpu().numpy()[0, 0]
        seg_mask = seg_mask.cpu().numpy()[0, 0]
        thermal_aura = thermal_aura.cpu().numpy()[0, 0]
        
        # Enhance contrast for visualization
        depth_map_uint8 = (depth_map * 255).astype(np.uint8)
        disparity_uint8 = (disparity * 255).astype(np.uint8)
        
        depth_map_enhanced = cv2.equalizeHist(depth_map_uint8)
        disparity_enhanced = cv2.equalizeHist(disparity_uint8)
        
        # Normalize back to [0, 1]
        depth_map_enhanced = depth_map_enhanced / 255.0
        disparity_enhanced = disparity_enhanced / 255.0
        
        # Create mutually reinforced fusion
        vis_gray = cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY)
        fused_img = (vis_gray.astype(float) * (1-seg_mask) + ir_img.astype(float) * seg_mask)
        fused_img = fused_img.astype(np.uint8)
        
        # Create color-enhanced visualizations
        thermal_viz = cv2.applyColorMap((thermal_aura * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        depth_viz = cv2.applyColorMap((depth_map_enhanced * 255).astype(np.uint8), cv2.COLORMAP_JET)
        disparity_viz = cv2.applyColorMap((disparity_enhanced * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        
        # Overlay thermal on IR
        alpha = 0.7
        ir_thermal = cv2.addWeighted(cv2.cvtColor(ir_img, cv2.COLOR_GRAY2BGR), 1-alpha, thermal_viz, alpha, 0)
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        plt.subplot(231)
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title('Visible Image')
        plt.axis('off')
        
        plt.subplot(232)
        plt.imshow(ir_img, cmap='gray')
        plt.title('IR Image')
        plt.axis('off')
        
        plt.subplot(233)
        plt.imshow(seg_mask, cmap='hot')
        plt.title('Segmentation Mask')
        plt.axis('off')
        
        plt.subplot(234)
        plt.imshow(cv2.cvtColor(depth_viz, cv2.COLOR_BGR2RGB))
        plt.title('Depth Map')
        plt.axis('off')
        
        plt.subplot(235)
        plt.imshow(cv2.cvtColor(disparity_viz, cv2.COLOR_BGR2RGB))
        plt.title('Disparity Map')
        plt.axis('off')
        
        plt.subplot(236)
        plt.imshow(cv2.cvtColor(ir_thermal, cv2.COLOR_BGR2RGB))
        plt.title('Thermal Aura')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"results_{i+1}.png"))
        plt.show()
        
        # Save a detailed view of the disparity matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(disparity_enhanced, cmap='plasma')
        plt.colorbar(label='Disparity Value')
        plt.title(f'Disparity Matrix - {vis_file} and {ir_file}')
        plt.savefig(os.path.join(output_dir, f"disparity_matrix_{i+1}.png"))
        plt.show()
        
        # Save the fused image
        cv2.imwrite(os.path.join(output_dir, f"fused_{i+1}.png"), fused_img)
        
        print(f"Processed and saved results for image pair {i+1}")

if __name__ == "__main__":
    process_all_images()
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Define a simplified HRNet model since we can't import the original
class SimpleHRNet(nn.Module):
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

class DepthEstimator:
    def __init__(self):
        # Check if pretrained HRNet weights exist
        pretrained_path = r"D:\DL\DepthEst\hrnetv2_w48_imagenet_pretrained.pth"
        if os.path.exists(pretrained_path):
            print(f"Found pretrained HRNet weights at {pretrained_path}")
            # Create our simplified model
            self.model = SimpleHRNet()
            
            # Load pretrained weights
            try:
                pretrained_weights = torch.load(pretrained_path)
                # Filter and adapt weights to our simplified model
                model_dict = self.model.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_weights.items() 
                                  if k in model_dict and v.shape == model_dict[k].shape}
                
                # Update model with pretrained weights
                if pretrained_dict:
                    print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained model")
                    model_dict.update(pretrained_dict)
                    self.model.load_state_dict(model_dict)
                else:
                    print("Could not load any weights from pretrained model (architecture mismatch)")
            except Exception as e:
                print(f"Error loading pretrained weights: {e}")
                print("Using randomly initialized model instead")
        else:
            print("Pretrained HRNet weights not found, using randomly initialized model")
            self.model = SimpleHRNet()
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        
        # Enhanced fusion and depth estimation layers
        self.fusion_layer1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.fusion_bn1 = nn.BatchNorm2d(512)
        self.fusion_layer2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.fusion_bn2 = nn.BatchNorm2d(256)
        self.depth_pred = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        
        # Segmentation module for Mutually Reinforcing Image Fusion and Segmentation
        self.segmentation_module = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=1)  # Foreground/background segmentation
        )
        
        if torch.cuda.is_available():
            self.fusion_layer1 = self.fusion_layer1.cuda()
            self.fusion_bn1 = self.fusion_bn1.cuda()
            self.fusion_layer2 = self.fusion_layer2.cuda()
            self.fusion_bn2 = self.fusion_bn2.cuda()
            self.depth_pred = self.depth_pred.cuda()
            self.segmentation_module = self.segmentation_module.cuda()

    def estimate_depth(self, vis_img, ir_img):
        with torch.no_grad():
            # Store original image dimensions for later resizing
            original_h, original_w = vis_img.shape[2], vis_img.shape[3]
            
            # Get features from model
            vis_feats = self.model(vis_img)
            ir_feats = self.model(ir_img)
            
            # Use multi-scale features for better results
            vis_feat = vis_feats[-1]  # Deepest features
            ir_feat = ir_feats[-1]
            
            # Generate segmentation masks for Mutually Reinforcing approach
            combined_feats = torch.cat([vis_feat, ir_feat], dim=1)
            seg_logits = self.segmentation_module(combined_feats)
            seg_mask = F.softmax(seg_logits, dim=1)[:, 1:2]  # Get foreground probability
            
            # Calculate disparity/correlation matrix
            vis_flat = vis_feat.view(vis_feat.size(0), vis_feat.size(1), -1)  # [B, C, H*W]
            ir_flat = ir_feat.view(ir_feat.size(0), ir_feat.size(1), -1)    # [B, C, H*W]
            
            # Compute similarity matrix
            similarity = torch.bmm(vis_flat.transpose(1, 2), ir_flat)  # [B, H*W, H*W]
            similarity = F.softmax(similarity * 20.0, dim=2)  # Temperature scaling
            
            # Convert to disparity map
            disparity_matrix = torch.matmul(vis_flat.transpose(1, 2), ir_flat) * 20.0
            disparity_matrix = F.softmax(disparity_matrix, dim=2)
            
            # Fix: Properly reshape the disparity map
            disparity_map, _ = torch.max(disparity_matrix, dim=2)  # Change to dim=2
            
            # Reshape disparity map to match feature map dimensions
            h, w = vis_feat.shape[2], vis_feat.shape[3]
            disparity_map = disparity_map.view(disparity_map.size(0), 1, h, w)
            
            # Use segmentation to enhance disparity map (now with proper dimensions)
            seg_mask_resized = F.interpolate(seg_mask, size=(h, w), 
                                           mode='bilinear', align_corners=True)
            
            # Apply segmentation mask to enhance disparity map
            disparity_map = disparity_map * (1.0 + 0.5 * seg_mask_resized)
            
            # Normalize disparity map for better visualization
            disparity_map = (disparity_map - disparity_map.min()) / (disparity_map.max() - disparity_map.min() + 1e-6)
            
            # Resize back to original image dimensions for visualization
            disparity_map_full = F.interpolate(disparity_map, size=(original_h, original_w), 
                                             mode='bilinear', align_corners=True)
            
            seg_mask_full = F.interpolate(seg_mask, size=(original_h, original_w), 
                                        mode='bilinear', align_corners=True)
            
            return disparity_map_full, disparity_matrix, seg_mask_full

def process_all_images():
    ir_path = r"D:\DL\M3FD_Fusion\ir"
    vis_path = r"D:\DL\M3FD_Fusion\vis"
    
    depth_est = DepthEstimator()
    
    ir_files = sorted(os.listdir(ir_path))[:3]
    vis_files = sorted(os.listdir(vis_path))[:3]
    
    for ir_file, vis_file in zip(ir_files, vis_files):
        ir_img = cv2.imread(os.path.join(ir_path, ir_file), cv2.IMREAD_GRAYSCALE)
        vis_img = cv2.imread(os.path.join(vis_path, vis_file))
        
        # Get dimensions for resizing later
        h, w = ir_img.shape[:2]
        
        ir_tensor = torch.from_numpy(ir_img).float().unsqueeze(0).unsqueeze(0) / 255.0
        vis_tensor = torch.from_numpy(vis_img).float().permute(2,0,1).unsqueeze(0) / 255.0
        
        if torch.cuda.is_available():
            ir_tensor = ir_tensor.cuda()
            vis_tensor = vis_tensor.cuda()
            
        # Get predictions
        disparity_map, disparity_matrix, seg_mask = depth_est.estimate_depth(vis_tensor, ir_tensor)
        
        # Convert to numpy
        disparity_map = disparity_map.cpu().numpy()[0, 0]
        disparity_matrix = disparity_matrix.cpu().numpy()[0]
        seg_mask = seg_mask.cpu().numpy()[0, 0]
        
        # Double-check shape matching before fusion
        # Ensure the sizes match exactly
        if seg_mask.shape != ir_img.shape:
            print(f"Resizing segmentation mask from {seg_mask.shape} to {ir_img.shape}")
            seg_mask = cv2.resize(seg_mask, (w, h))
        
        # Create mutually reinforced fused image using segmentation
        vis_gray = cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY)
        fused_img = (vis_gray.astype(float) * (1-seg_mask) + 
                     ir_img.astype(float) * seg_mask)
        fused_img = fused_img.astype(np.uint8)
        
        # Plot results with segmentation
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
        plt.title(f'Segmentation Mask {seg_mask.shape}')
        plt.axis('off')
        
        plt.subplot(234)
        plt.imshow(disparity_map, cmap='jet')
        plt.title(f'Disparity Map {disparity_map.shape}')
        plt.axis('off')
        
        plt.subplot(235)
        plt.imshow(disparity_matrix, cmap='viridis')
        plt.title('Disparity Matrix')
        plt.axis('off')
        
        plt.subplot(236)
        plt.imshow(fused_img, cmap='gray')
        plt.title('Mutually Reinforced Fusion')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    process_all_images()
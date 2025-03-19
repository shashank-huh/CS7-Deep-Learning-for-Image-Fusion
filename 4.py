import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Define StereoHRNet model architecture with enhanced disparity estimation
class StereoHRNet(nn.Module):
    def __init__(self, config):
        """
        Stereo matching network using HRNet backbone with focus on disparity estimation
        """
        super(StereoHRNet, self).__init__()
        
        # Enhanced feature extraction for visible and IR branches
        self.vis_features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((64, 64))
        )
        
        self.ir_features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((64, 64))
        )

        # Multi-scale segmentation for feature enhancement
        self.segmentation = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 2, kernel_size=1)
            ) for _ in range(4)  # 4 scales for better multi-scale features
        ])
        
        # Enhanced fusion attention for disparity estimation
        self.fusion_attention = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Refined cross-modal attention
        self.cross_attention = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # Additional disparity refinement module
        self.disparity_refine = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # Changed input channels from 512 to 256
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

    def compute_disparity_matrix(self, vis_feats, ir_feats):
        """
        Compute refined disparity matrix with enhanced cross-modal attention
        """
        b, c, h, w = vis_feats.shape
        
        # Apply refinement before correlation
        vis_feats = self.disparity_refine(vis_feats)
        ir_feats = self.disparity_refine(ir_feats)
        
        vis_flat = vis_feats.view(b, -1, h*w)
        ir_flat = ir_feats.view(b, -1, h*w)
        
        # Multi-scale correlation with learnable temperature and normalization
        disparity_matrix = torch.matmul(vis_flat.transpose(1,2), ir_flat)
        disparity_matrix = disparity_matrix / (torch.norm(vis_flat, dim=1, keepdim=True) * 
                                             torch.norm(ir_flat, dim=1, keepdim=True).transpose(1,2))
        
        # Apply spatial attention
        disparity_matrix = F.softmax(disparity_matrix, dim=-1)
        
        return disparity_matrix.view(b, h*w, h, w)

    def forward(self, vis, ir):
        """
        Forward pass with enhanced disparity estimation
        Args:
            vis: Visible image tensor [B,3,H,W]
            ir: IR image tensor [B,1,H,W]
        Returns:
            multi-scale segmentation masks, disparity matrix, attention maps
        """
        # Extract enhanced multi-scale features
        vis_feats = self.vis_features(vis)  # [B,256,H,W]
        ir_feats = self.ir_features(ir)     # [B,256,H,W]
        
        # Compute refined cross-modal attention and disparity
        cross_attn = self.cross_attention(torch.cat([vis_feats, ir_feats], dim=1))
        
        # Apply attention before disparity computation
        vis_attended = vis_feats * cross_attn[:,0:1]  # [B,256,H,W]
        ir_attended = ir_feats * cross_attn[:,1:2]    # [B,256,H,W]
        
        # Enhanced feature fusion
        fused_feats = torch.cat([vis_attended, ir_attended], dim=1)  # [B,512,H,W]
        attention_weights = self.fusion_attention(fused_feats)
        enhanced_feats = fused_feats * attention_weights
        
        # Multi-scale segmentation for feature enhancement
        seg_outputs = []
        prev_feat = None
        for seg_branch in self.segmentation:
            if prev_feat is not None:
                enhanced_feats = enhanced_feats + prev_feat
            seg_mask = seg_branch(enhanced_feats)
            seg_outputs.append(seg_mask)
            prev_feat = enhanced_feats
            
        # Use segmentation feedback to refine disparity estimation
        seg_attention = F.softmax(seg_outputs[-1], dim=1)[:,1:2]
        refined_feats = enhanced_feats * seg_attention
        
        # Final disparity computation with refined features
        # Split refined_feats back into vis and ir components
        refined_vis = refined_feats[:,:256,...]  # Take first half of channels
        refined_ir = refined_feats[:,256:,...]   # Take second half of channels
        
        disparity_matrix = self.compute_disparity_matrix(vis_attended + refined_vis,
                                                       ir_attended + refined_ir)
        
        return seg_outputs, disparity_matrix, cross_attn

def process_images(vis_path, ir_path):
    """
    Process image pairs with enhanced disparity visualization
    """
    model = StereoHRNet(None)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    
    ir_files = sorted(os.listdir(ir_path))[:3]
    vis_files = sorted(os.listdir(vis_path))[:3]
    
    for ir_file, vis_file in zip(ir_files, vis_files):
        # Load and preprocess
        ir_img = cv2.imread(os.path.join(ir_path, ir_file), cv2.IMREAD_GRAYSCALE)
        vis_img = cv2.imread(os.path.join(vis_path, vis_file))
        
        # Normalize and convert to tensors
        ir_tensor = torch.from_numpy(ir_img).float().unsqueeze(0).unsqueeze(0) / 255.0
        vis_tensor = torch.from_numpy(vis_img).float().permute(2,0,1).unsqueeze(0) / 255.0
        
        if torch.cuda.is_available():
            ir_tensor = ir_tensor.cuda()
            vis_tensor = vis_tensor.cuda()
            
        # Get predictions
        with torch.no_grad():
            seg_outputs, disparity_matrix, cross_attn = model(vis_tensor, ir_tensor)
        
        # Convert to numpy
        seg_mask = F.softmax(seg_outputs[-1], dim=1).cpu().numpy()[0,1]
        disparity_matrix = disparity_matrix.cpu().numpy()[0,0]
        cross_attn = cross_attn.cpu().numpy()[0]
        
        # Plot results
        plt.figure(figsize=(15,10))
        
        plt.subplot(231)
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title('Visible Image')
        plt.axis('off')
        
        plt.subplot(232)
        plt.imshow(ir_img, cmap='gray')
        plt.title('IR Image')
        plt.axis('off')
        
        plt.subplot(233)
        plt.imshow(disparity_matrix, cmap='magma')
        plt.colorbar()
        plt.title('Disparity Matrix')
        plt.axis('off')
        
        plt.subplot(234)
        plt.imshow(seg_mask, cmap='jet')
        plt.title('Segmentation Mask')
        plt.axis('off')
        
        plt.subplot(235)
        plt.imshow(cross_attn[0], cmap='viridis')
        plt.title('Vis Attention')
        plt.axis('off')
        
        plt.subplot(236)
        plt.imshow(cross_attn[1], cmap='viridis')
        plt.title('IR Attention')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    ir_path = r"D:\DL\M3FD_Fusion\ir"
    vis_path = r"D:\DL\M3FD_Fusion\vis" 
    process_images(vis_path, ir_path)

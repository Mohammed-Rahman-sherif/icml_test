from __future__ import annotations
import torch
import torch.nn as nn
from hgtmodel import HGTImageFeatureExtractor
# ==========================================
# PART 2: The Integrated Wrapper (Replaces Graph Builder)
# ==========================================
class IntegratedHGTModel(nn.Module):
    """
    Wraps the CLIP backbone and the HGT model.
    Builds the graph structure on the GPU on-the-fly.
    """
    def __init__(self, clip_model, hgt_model, num_classes, num_patches=18):
        super().__init__()
        self.clip_model = clip_model  # HuggingFace CLIP Model (with LoRA already injected)
        self.hgt_model = hgt_model # <--- Use the object passed in arguments    # The class defined above
        self.num_patches = num_patches
        self.num_classes = num_classes
        
        # --- Pre-compute BASE Edge Indices (For 1 single graph) ---
        # We define the structure for ONE image here. 
        # The forward pass will mathematically expand this for the whole batch.

        # 1. Intra-Patch Edges (vit <-> vit) [Dense 18x18]
        s = torch.arange(num_patches).repeat_interleave(num_patches)
        d = torch.arange(num_patches).repeat(num_patches)
        mask = s != d  # Remove self-loops
        self.register_buffer('base_vit_intra', torch.stack([s[mask], d[mask]], dim=0))

        # 2. Visual-to-Text Edges (vit -> text) [Dense 18 x num_classes]
        # src (vit): 0..17 repeated num_classes times
        # dst (text): 0..num_classes repeated
        s_vt = torch.arange(num_patches).repeat_interleave(num_classes)
        d_vt = torch.arange(num_classes).repeat(num_patches)
        self.register_buffer('base_vit_text', torch.stack([s_vt, d_vt], dim=0))
        
        # 3. Text-to-Visual (Inverse of above)
        self.register_buffer('base_text_vit', torch.stack([d_vt, s_vt], dim=0))

    def create_batch_graph(self, batch_size, device):
        """
        Generates the batch vector and edge indices for the entire batch at once.
        """
        # --- 1. Create Batch Vectors ---
        # Output: [0,0,0... 1,1,1... ]
        vit_batch_vec = torch.arange(batch_size, device=device).repeat_interleave(self.num_patches)
        text_batch_vec = torch.arange(batch_size, device=device).repeat_interleave(self.num_classes)
        
        # --- 2. Create Batch Edges ---
        # Helper to expand base edges to batch size
        def expand_edges(base_edges, src_dim, dst_dim):
            # base_edges: [2, E]
            num_edges = base_edges.shape[1]
            
            # Create offsets [0, 1, ... B-1]
            offsets = torch.arange(batch_size, device=device).view(-1, 1)
            
            # Calculate how much to shift indices for each item in batch
            src_shift = offsets * src_dim
            dst_shift = offsets * dst_dim
            
            # Repeat base edges for batch
            src_base = base_edges[0].unsqueeze(0).expand(batch_size, -1) # [B, E]
            dst_base = base_edges[1].unsqueeze(0).expand(batch_size, -1) # [B, E]
            
            # Add shifts and flatten
            src_final = (src_base + src_shift).flatten()
            dst_final = (dst_base + dst_shift).flatten()
            
            return torch.stack([src_final, dst_final], dim=0)

        # Apply expansion
        edge_intra = expand_edges(self.base_vit_intra, self.num_patches, self.num_patches)
        edge_v2t   = expand_edges(self.base_vit_text,  self.num_patches, self.num_classes)
        edge_t2v   = expand_edges(self.base_text_vit,  self.num_classes, self.num_patches)
        
        return vit_batch_vec, text_batch_vec, edge_intra, edge_v2t, edge_t2v

    def forward(self, patch_images, text_features_prototype):
        """
        Args:
            patch_images: [Batch, 18, 3, 224, 224] (Raw pixel tensors)
            text_features_prototype: [num_classes, feature_dim] (Pre-computed text feats)
        """
        B, N, C, H, W = patch_images.shape
        
        # --- 1. Feature Extraction (CLIP + LoRA) ---
        # Flatten patches: [B*18, 3, 224, 224]
        flat_images = patch_images.view(-1, C, H, W) 
        
        # Use HuggingFace CLIP vision model
        # Gradients from LoRA will flow through here!
        vision_outputs = self.clip_model.vision_model(pixel_values=flat_images)
        vit_features = vision_outputs.pooler_output # [B*18, 768]
        
        # Prepare Text Features (Replicate prototype for batch)
        # [num_classes, dim] -> [B * num_classes, dim]
        text_features_batch = text_features_prototype.repeat(B, 1)
        
        # --- 2. Graph Construction (On GPU) ---
        vit_batch, text_batch, e_intra, e_v2t, e_t2v = self.create_batch_graph(B, device=vit_features.device)
        
        # --- 3. Pack for HGT ---
        x_dict = {
            'vit': vit_features,
            'text': text_features_batch
        }
        
        edge_index_dict = {
            ('vit', 'intra_patch', 'vit'): e_intra,
            ('vit', 'visual_to_text', 'text'): e_v2t,
            ('text', 'text_to_visual', 'vit'): e_t2v
        }
        
        batch_dict = {
            'vit': vit_batch,
            'text': text_batch
        }
        
        # --- 4. Run HGT ---
        return self.hgt_model(x_dict, edge_index_dict, batch_dict)
from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, global_add_pool, TopKPooling

# ==========================================
# PART 1: Your Original HGT Feature Extractor
# ==========================================
class HGTImageFeatureExtractor(nn.Module):
    """
    Your original HGT model. No changes to logic, just encapsulated here.
    """
    def __init__(self, node_types, edge_types, input_dims, hidden_channels,
                 hgt_num_heads, hgt_num_layers, dropout_rate,
                 transformer_nhead, transformer_num_layers,
                 transformer_ff_multiplier, transformer_activation, shots,
                 pooling_ratio: float):
        super().__init__()

        self.node_types = node_types
        self.visual_node_types = [nt for nt in node_types if nt != 'text']
        self.edge_types = edge_types
        self.metadata = (self.node_types, self.edge_types)
        self.hidden_channels = hidden_channels
        self.num_hgt_layers = hgt_num_layers
        self.shots = shots

        # --- Input Projection ---
        self.input_proj = nn.ModuleDict()
        for node_type in self.node_types:
            self.input_proj[node_type] = nn.Linear(input_dims[node_type], hidden_channels)

        # --- Transformer Encoders ---
        self.transformer_encoders = nn.ModuleDict()
        for node_type in self.node_types:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_channels, nhead=transformer_nhead,
                dim_feedforward=hidden_channels * transformer_ff_multiplier,
                dropout=dropout_rate, activation=transformer_activation, batch_first=True
            )
            self.transformer_encoders[node_type] = nn.TransformerEncoder(
                encoder_layer, num_layers=transformer_num_layers
            )

        # --- HGT Layers ---
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(self.num_hgt_layers):
            self.convs.append(HGTConv(hidden_channels, hidden_channels, self.metadata, hgt_num_heads))
            norm_dict = nn.ModuleDict()
            for node_type in self.node_types:
                norm_dict[node_type] = nn.LayerNorm(hidden_channels)
            self.norms.append(norm_dict)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # --- Pooling ---
        self.topk_pools = nn.ModuleDict()
        for node_type in self.visual_node_types:
            self.topk_pools[node_type] = TopKPooling(hidden_channels, ratio=pooling_ratio)
        
    def forward(self, x_dict, edge_index_dict, batch_dict):
        # 1. Input Projection
        projected_x_dict = {}
        for node_type, x_features in x_dict.items():
            projected_x_dict[node_type] = self.input_proj[node_type](x_features)

        # 2. Transformer Encoder
        transformed_x_dict = {}
        for node_type, x_features in projected_x_dict.items():
            # Simply pass through if no complicated batching needed for Transformer
            # (Simplified for brevity, assuming inputs are already aligned)
            transformed_x_dict[node_type] = self.transformer_encoders[node_type](x_features.unsqueeze(1)).squeeze(1)

        # 3. HGT Convolutions
        current_x_dict = transformed_x_dict
        for conv, norm_dict in zip(self.convs, self.norms):
            x_in = current_x_dict
            x_out = conv(current_x_dict, edge_index_dict)
            for node_type in x_in.keys():
                out = self.dropout(norm_dict[node_type](x_out[node_type]).relu())
                current_x_dict[node_type] = x_in[node_type] + out
        
        # 4. Pooling (Visual Only)
        pooled_visual_features = []    
        for node_type in self.visual_node_types:
            features = current_x_dict[node_type]
            # Use intra-patch edges for pooling if available
            intra_edge_type = (node_type, 'intra_patch', node_type)
            edge_index = edge_index_dict.get(intra_edge_type, torch.empty(2, 0, device=features.device, dtype=torch.long))

            if edge_index.numel() == 0:
                selected_features = features
                selected_batch = batch_dict[node_type]
            else:
                selected_features, _, _, selected_batch, _, _ = self.topk_pools[node_type](
                    x=features, edge_index=edge_index, batch=batch_dict[node_type]
                )
            pooled_visual_features.append(global_add_pool(selected_features, selected_batch))
        
        graph_visual_feature = pooled_visual_features[0]
        all_updated_text_features = current_x_dict['text']
        
        return graph_visual_feature, all_updated_text_features



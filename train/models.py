import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GlobalAttention, global_max_pool, global_mean_pool

class HE_ResGATConv(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_classes=1879, dropout=0.3, num_layers=4, heads=8, use_legacy_arch=False):
        super().__init__()
        self.dropout_rate = dropout
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.use_legacy_arch = use_legacy_arch
        
        # Legacy Mode Implementation
        if self.use_legacy_arch:
            # 1. Input Projection (Linear 1536 -> 256)
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            
            # 2. GAT Layers with BatchNorm
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            for i in range(self.num_layers):
                self.convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=self.heads, concat=False, edge_dim=1))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
                
            # 3. Classifier (Input 512 -> Output num_classes)
            # Assuming concatenation of Max + Mean pooling (256 + 256 = 512)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(512, num_classes)
            )
            return

        # Standard Mode Implementation (Simplified & Robust)
        # 1. Input Projection (Enhanced with dimensionality reduction)
        # Give model more operating space: input_dim -> reduced_dim -> hidden_dim
        reduction_dim = max(hidden_dim, input_dim // 3)
        
        if reduction_dim != hidden_dim and input_dim > hidden_dim:
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, reduction_dim),
                nn.BatchNorm1d(reduction_dim),
                nn.PReLU(),
                nn.Dropout(dropout),
                nn.Linear(reduction_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.PReLU()
            )
        else:
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.PReLU()
            )
        
        # 2. ResGAT Layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(self.num_layers):
            self.convs.append(GATv2Conv(
                hidden_dim, hidden_dim // self.heads, 
                heads=self.heads, concat=True, edge_dim=1,
                dropout=dropout
            ))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # 3. Pooling (Mean + Max)
        # Output dim = hidden_dim * 2
        
        # 4. Classifier
        # JK-Connection: Concatenate output of all layers
        # Pooling output size: (hidden_dim * num_layers) * 2
        self.classifier_input_dim = hidden_dim * num_layers * 2
        
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self._init_weights()

    def _init_weights(self):
        # Initialize final layer bias for imbalanced classification
        # pi = 0.01
        # bias = -log((1-pi)/pi) = -log(99) = -4.59
        if isinstance(self.classifier[-1], nn.Linear):
            nn.init.constant_(self.classifier[-1].bias, -4.59)

    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        # 兼容 PyG Data 格式
        if hasattr(x, 'x') and hasattr(x, 'edge_index'):
            data = x
            x, edge_index = data.x, data.edge_index
            edge_attr = getattr(data, 'edge_attr', None)
            batch = getattr(data, 'batch', torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        elif batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if edge_attr is not None:
             # Transform edge distance to similarity/weight if needed
             # Assuming edge_attr is distance
             edge_attr = torch.exp(-edge_attr)

        if self.use_legacy_arch:
            x = self.input_proj(x)
            for conv, bn in zip(self.convs, self.bns):
                x_in = x
                x = conv(x, edge_index, edge_attr=edge_attr)
                x = bn(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
                x = x + x_in
            
            x1 = global_max_pool(x, batch)
            x2 = global_mean_pool(x, batch)
            x = torch.cat([x1, x2], dim=1)
            return self.classifier(x)

        # Standard Mode Forward
        # 1. Input Projection
        x = self.input_proj(x)
        
        # Store layer outputs for JK-Connection
        layer_outputs = []
        
        # 2. ResGAT Blocks
        for conv, bn in zip(self.convs, self.bns):
            x_in = x
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            x = F.relu(x) # or PReLU if used in layer
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = x + x_in # Residual Connection
            layer_outputs.append(x)

        # 3. Pooling with JK (Concatenate all layers)
        # layer_outputs: [x1, x2, x3, x4]
        # We pool each xi separately and then cat, or cat then pool (equivalent for global pool)
        
        pooled_outputs = []
        for xi in layer_outputs:
            xi_max = global_max_pool(xi, batch)
            xi_mean = global_mean_pool(xi, batch)
            pooled_outputs.append(xi_mean)
            pooled_outputs.append(xi_max)
            
        # Concatenate: [mean1, max1, mean2, max2, ...]
        x = torch.cat(pooled_outputs, dim=1) 
        
        # 4. Classifier
        out = self.classifier(x)
        return out


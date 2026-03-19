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
        
        if self.use_legacy_arch:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            for i in range(self.num_layers):
                self.convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=self.heads, concat=False, edge_dim=1))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
                
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(512, num_classes)
            )
            return

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
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(self.num_layers):
            self.convs.append(GATv2Conv(
                hidden_dim, hidden_dim // self.heads, 
                heads=self.heads, concat=True, edge_dim=1,
                dropout=dropout
            ))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

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
        if isinstance(self.classifier[-1], nn.Linear):
            nn.init.constant_(self.classifier[-1].bias, -4.59)

    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        if hasattr(x, 'x') and hasattr(x, 'edge_index'):
            data = x
            x, edge_index = data.x, data.edge_index
            edge_attr = getattr(data, 'edge_attr', None)
            batch = getattr(data, 'batch', torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        elif batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if edge_attr is not None:
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

        x = self.input_proj(x)
        
        layer_outputs = []
        
        for conv, bn in zip(self.convs, self.bns):
            x_in = x
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = x + x_in
            layer_outputs.append(x)

        pooled_outputs = []
        for xi in layer_outputs:
            xi_max = global_max_pool(xi, batch)
            xi_mean = global_mean_pool(xi, batch)
            pooled_outputs.append(xi_mean)
            pooled_outputs.append(xi_max)
            
        x = torch.cat(pooled_outputs, dim=1) 
        
        out = self.classifier(x)
        return out


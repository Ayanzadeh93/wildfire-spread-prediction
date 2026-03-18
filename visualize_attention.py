import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GATConv
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Re-implement architecture classes for loading
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

def get_grid_topology(H, W, device):
    src_nodes, dst_nodes = [], []
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for r in range(H):
        for c in range(W):
            curr_node = r * W + c
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    neighbor_node = nr * W + nc
                    src_nodes.append(curr_node)
                    dst_nodes.append(neighbor_node)
    return torch.tensor([src_nodes, dst_nodes], dtype=torch.long, device=device)

class AttentionHybridFireGNN(nn.Module):
    def __init__(self, in_channels=12, hidden_dim=64, dropout=0.3):
        super(AttentionHybridFireGNN, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.enc3 = nn.Sequential(nn.Conv2d(64, hidden_dim, 3, 1, 1), nn.BatchNorm2d(hidden_dim), nn.ReLU(), nn.MaxPool2d(2))
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
        self.gat3 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
        self.up1 = nn.ConvTranspose2d(hidden_dim, 64, 2, 2)
        self.att1 = AttentionGate(64, 64, 32)
        self.conv_up1 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.att2 = AttentionGate(32, 32, 16)
        self.conv_up2 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU())
        self.up3 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.final = nn.Conv2d(16, 1, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        B, C_lat, H_lat, W_lat = e3.shape
        x_flat = e3.permute(0, 2, 3, 1).reshape(B * H_lat * W_lat, C_lat)
        single_edges = get_grid_topology(H_lat, W_lat, x.device)
        edge_indices = [single_edges + i * (H_lat * W_lat) for i in range(B)]
        batched_edges = torch.cat(edge_indices, dim=1)
        g = F.elu(self.gat1(x_flat, batched_edges))
        g = F.elu(self.gat2(g, batched_edges))
        g = F.elu(self.gat3(g, batched_edges)) + x_flat
        x_gnn = g.reshape(B, H_lat, W_lat, C_lat).permute(0, 3, 1, 2)
        g1 = self.up1(x_gnn)
        x1 = self.att1(g=g1, x=e2)
        d1 = self.conv_up1(torch.cat([g1, x1], dim=1))
        g2 = self.up2(d1)
        x2 = self.att2(g=g2, x=e1)
        d2 = self.conv_up2(torch.cat([g2, x2], dim=1))
        return self.final(self.up3(d2))

class WildfireDataset(Dataset):
    def __init__(self, data_dir, split='test'):
        split_dir = os.path.join(data_dir, split)
        self.files = sorted([os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith('.npy')])
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        x, y = torch.tensor(data[:-1, ...], dtype=torch.float32), torch.tensor(data[-1:, ...], dtype=torch.float32)
        m, s = x.mean(dim=(1, 2), keepdim=True), x.std(dim=(1, 2), keepdim=True)
        return (x - m) / (s + 1e-6), y

# Load model and visualize
model = AttentionHybridFireGNN().to(device)
model.load_state_dict(torch.load('best_attention_model.pth', weights_only=True))
model.eval()

test_ds = WildfireDataset(r'c:\Tim\Wildfire_physics\processed_data_full', split='test')
loader = DataLoader(test_ds, batch_size=1)

# Find a sample with fire
found_fire = False
for x, y in loader:
    if y.sum() > 10:
        x, y = x.to(device), y.to(device)
        prob = torch.sigmoid(model(x))
        found_fire = True
        break

if found_fire:
    input_fire = x[0, 11, :, :].cpu().numpy()
    target_fire = y[0, 0, :, :].cpu().numpy()
    pred_prob = prob[0, 0, :, :].detach().cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(input_fire, cmap='Reds'); axes[0].set_title("Input: Prev-Day Fire"); axes[0].axis('off')
    axes[1].imshow(target_fire, cmap='Reds'); axes[1].set_title("Target: Next-Day Fire"); axes[1].axis('off')
    im = axes[2].imshow(pred_prob, cmap='jet', vmin=0, vmax=1); axes[2].set_title("Attention U-Net Prob"); axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig('prediction_attention.png', dpi=150)
    print("Saved prediction_attention.png")

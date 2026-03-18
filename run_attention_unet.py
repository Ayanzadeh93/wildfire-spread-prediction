import os
import glob
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GATConv
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, average_precision_score, jaccard_score, precision_recall_fscore_support, roc_curve, precision_recall_curve

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================
# DATA PROCESSING AND SPLITTING (Reusing same data)
# ============================================
OUTPUT_DIR = r'c:\Tim\Wildfire_physics\processed_data_full'

class WildfireDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Directory {split_dir} not found. Please run run_unet.py first to prepare data.")
        self.files = sorted([os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith('.npy')])
        print(f"Loaded {len(self.files)} samples for {split} split")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        x = torch.tensor(data[:-1, :, :], dtype=torch.float32)
        y = torch.tensor(data[-1:, :, :], dtype=torch.float32)
        
        # Z-score normalization per sample
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True)
        x = (x - mean) / (std + 1e-6)
        
        return x, y

# ============================================
# ATTENTION U-NET COMPONENTS
# ============================================

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
    src_nodes = []
    dst_nodes = []
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               ( 0, -1),          ( 0, 1),
               ( 1, -1), ( 1, 0), ( 1, 1)]
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
        
        # CNN Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1), 
            nn.BatchNorm2d(32), nn.ReLU(), 
            nn.MaxPool2d(2)  # 64->32
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1), 
            nn.BatchNorm2d(64), nn.ReLU(), 
            nn.MaxPool2d(2)  # 32->16
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, hidden_dim, 3, 1, 1), 
            nn.BatchNorm2d(hidden_dim), nn.ReLU(), 
            nn.MaxPool2d(2)  # 16->8
        )

        # GAT Bottleneck
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
        self.gat3 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout)

        # Attention Decoder
        self.up1 = nn.ConvTranspose2d(hidden_dim, 64, 2, 2)  # 8->16
        self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.conv_up1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64), nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(64, 32, 2, 2)  # 16->32
        self.att2 = AttentionGate(F_g=32, F_l=32, F_int=16)
        self.conv_up2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32), nn.ReLU()
        )

        self.up3 = nn.ConvTranspose2d(32, 16, 2, 2)  # 32->64
        self.final = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)  # 32 channels, 32x32
        e2 = self.enc2(e1) # 64 channels, 16x16
        e3 = self.enc3(e2) # 64 channels, 8x8
        
        B, C_lat, H_lat, W_lat = e3.shape
        x_flat = e3.permute(0, 2, 3, 1).reshape(B * H_lat * W_lat, C_lat)
        single_edges = get_grid_topology(H_lat, W_lat, x.device)
        edge_indices = [single_edges + i * (H_lat * W_lat) for i in range(B)]
        batched_edges = torch.cat(edge_indices, dim=1)
        
        g = F.elu(self.gat1(x_flat, batched_edges))
        g = F.elu(self.gat2(g, batched_edges))
        g = F.elu(self.gat3(g, batched_edges)) + x_flat
        x_gnn = g.reshape(B, H_lat, W_lat, C_lat).permute(0, 3, 1, 2)

        # Decoder with Attention
        g1 = self.up1(x_gnn)
        x1 = self.att1(g=g1, x=e2)
        d1 = self.conv_up1(torch.cat([g1, x1], dim=1))

        g2 = self.up2(d1)
        x2 = self.att2(g=g2, x=e1)
        d2 = self.conv_up2(torch.cat([g2, x2], dim=1))

        out = self.final(self.up3(d2))
        return out

# ============================================
# EVALUATION & VISUALIZATION (Adapted for Attention)
# ============================================
def find_optimal_threshold(model, val_loader, device):
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            probs = torch.sigmoid(model(x))
            all_probs.extend(probs.cpu().numpy().flatten())
            all_targets.extend((y > 0.0).float().cpu().numpy().flatten())
    y_true, y_prob = np.array(all_targets), np.array(all_probs)
    thresholds = np.arange(0.2, 0.7, 0.05)
    best_f1, best_thresh = 0, 0.5
    for thresh in thresholds:
        f1 = f1_score(y_true, (y_prob > thresh).astype(float), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    print(f"Optimal threshold: {best_thresh:.2f} (F1={best_f1:.3f})")
    return best_thresh

def evaluate_on_test_set(model, test_loader, device, threshold):
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            probs = torch.sigmoid(model(x))
            all_probs.extend(probs.cpu().numpy().flatten())
            all_targets.extend((y > 0.0).float().cpu().numpy().flatten())
    y_true, y_prob = np.array(all_targets), np.array(all_probs)
    y_pred = (y_prob > threshold).astype(float)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    iou = jaccard_score(y_true, y_pred, zero_division=0)
    auprc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    print("\n" + "="*70)
    print(" ATTENTION U-NET TEST RESULTS")
    print("="*70)
    print(f"IoU: {iou:.3f} | Recall: {recall:.3f} | Prec: {precision:.3f} | F1: {f1:.3f} | AUPRC: {auprc:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")
    return iou, recall, precision, f1, auprc, roc_auc

# ============================================
# MAIN TRAINING SCRIPT
# ============================================
def main():
    train_dataset = WildfireDataset(OUTPUT_DIR, split='train')
    val_dataset = WildfireDataset(OUTPUT_DIR, split='val')
    test_dataset = WildfireDataset(OUTPUT_DIR, split='test')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    model = AttentionHybridFireGNN(in_channels=12, hidden_dim=64, dropout=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.0]).to(device))
    NUM_EPOCHS = 100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    print("\nTRAINING: Attention Hybrid ResGNN-UNet")
    for epoch in range(NUM_EPOCHS):
        model.train()
        t_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), (y > 0).float().to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        
        avg_t = t_loss / len(train_loader)
        train_losses.append(avg_t)

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                v_loss += criterion(model(x.to(device)), (y.to(device) > 0).float()).item()
        avg_v = v_loss / len(val_loader)
        val_losses.append(avg_v)
        scheduler.step()
        
        if avg_v < best_val_loss:
            best_val_loss = avg_v
            torch.save(model.state_dict(), 'best_attention_model.pth')
        print(f"Epoch {epoch+1:2d} | Train: {avg_t:.4f} | Val: {avg_v:.4f}")

    # Final Comparison
    model.load_state_dict(torch.load('best_attention_model.pth', weights_only=True))
    thresh = find_optimal_threshold(model, val_loader, device)
    res = evaluate_on_test_set(model, test_loader, device, thresh)
    
    # Simple result save for comparison
    with open('attention_results.txt', 'w') as f:
        f.write(f"IoU,Recall,Precision,F1,AUPRC,ROCAUC\n")
        f.write(",".join([f"{x:.4f}" for x in res]))

if __name__ == '__main__':
    main()

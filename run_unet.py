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
# DATA PROCESSING AND SPLITTING
# ============================================
INPUT_DIR = r'c:\Tim\Wildfire_physics\archive'
OUTPUT_DIR = r'c:\Tim\Wildfire_physics\processed_data_full'

# Create directories
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# Define features for Next-Day Wildfire dataset
IMG_SHAPE = [64, 64]
features_dict = {
    'elevation': tf.io.FixedLenFeature(IMG_SHAPE, tf.float32),
    'th': tf.io.FixedLenFeature(IMG_SHAPE, tf.float32),
    'vs': tf.io.FixedLenFeature(IMG_SHAPE, tf.float32),
    'tmmn': tf.io.FixedLenFeature(IMG_SHAPE, tf.float32),
    'tmmx': tf.io.FixedLenFeature(IMG_SHAPE, tf.float32),
    'sph': tf.io.FixedLenFeature(IMG_SHAPE, tf.float32),
    'pr': tf.io.FixedLenFeature(IMG_SHAPE, tf.float32),
    'pdsi': tf.io.FixedLenFeature(IMG_SHAPE, tf.float32),
    'NDVI': tf.io.FixedLenFeature(IMG_SHAPE, tf.float32),
    'erc': tf.io.FixedLenFeature(IMG_SHAPE, tf.float32),
    'population': tf.io.FixedLenFeature(IMG_SHAPE, tf.float32),
    'PrevFireMask': tf.io.FixedLenFeature(IMG_SHAPE, tf.float32),
    'FireMask': tf.io.FixedLenFeature(IMG_SHAPE, tf.float32)
}

def parse_tfrecord(example_proto):
    parsed_features = tf.io.parse_single_example(example_proto, features_dict)
    inputs = tf.stack([
        parsed_features['elevation'], parsed_features['th'], parsed_features['vs'],
        parsed_features['tmmn'], parsed_features['tmmx'], parsed_features['sph'],
        parsed_features['pr'], parsed_features['pdsi'], parsed_features['NDVI'],
        parsed_features['erc'], parsed_features['population'], parsed_features['PrevFireMask']
    ], axis=0)
    target = tf.expand_dims(parsed_features['FireMask'], axis=0)
    # Combine inputs and target for easy saving
    return tf.concat([inputs, target], axis=0)

def prepare_data():
    if len(glob.glob(os.path.join(OUTPUT_DIR, 'train', '*.npy'))) > 0:
        print(f"Data already processed in {OUTPUT_DIR}. Skipping extraction.")
        return

    print(f"Searching for TFRecords in {INPUT_DIR}...")
    tf_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.tfrecord')))
    print(f"Found {len(tf_files)} TFRecord files.")

    if not tf_files:
        print("Error: No TFRecord files found.")
        return

    # To process everything faster, we'll parse them into lists and save.
    dataset = tf.data.TFRecordDataset(tf_files)
    dataset = dataset.map(parse_tfrecord)
    
    all_data = []
    print("Extracting records from TFRecord format...")
    # Add a counter just to show progress
    for i, data in enumerate(dataset):
        all_data.append(data.numpy()) # convert to numpy here
        if (i+1) % 1000 == 0:
            print(f"  Extracted {i+1} samples...")
    
    total = len(all_data)
    print(f"Total samples loaded: {total}")
    
    # 75% Train, 10% Val, 15% Test
    train_end = int(0.75 * total)
    val_end = int(0.85 * total)
    
    splits = {
        'train': all_data[:train_end],
        'val': all_data[train_end:val_end],
        'test': all_data[val_end:]
    }
    
    for split_name, split_data in splits.items():
        print(f"Saving {split_name}: {len(split_data)} samples to {OUTPUT_DIR}\\{split_name}...")
        for i, data in enumerate(split_data):
            np.save(os.path.join(OUTPUT_DIR, split_name, f'sample_{i}.npy'), data)

    print("\\n=== DATA SPLIT COMPLETE ===")

# ============================================
# DATASET CLASS
# ============================================
class WildfireDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        split_dir = os.path.join(data_dir, split)
        self.files = sorted([os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith('.npy')])
        print(f"Loaded {len(self.files)} samples for {split} split")

        if len(self.files) > 0 and split == 'train':
            self.preliminary_analysis()

    def __len__(self):
        return len(self.files)

    def preliminary_analysis(self):
        print("\\n=== PRELIMINARY DATASET ANALYSIS ===")
        # Look at the first 100 files to get some stats
        num_samples = min(100, len(self.files))
        fire_pixels = 0
        total_pixels = 0
        for i in range(num_samples):
            data = np.load(self.files[i])
            y = data[-1:, :, :]
            fire_pixels += np.sum(y > 0)
            total_pixels += y.size
        print(f"Sampled {num_samples} training tiles.")
        print(f"Total target pixels: {total_pixels}")
        print(f"Positive Fire target pixels: {fire_pixels}")
        print(f"Class Imbalance Ratio (Fire/Total): {fire_pixels/total_pixels:.4%}")
        print("======================================\\n")


    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        x = torch.tensor(data[:-1, :, :], dtype=torch.float32)
        y = torch.tensor(data[-1:, :, :], dtype=torch.float32)
        
        # Z-score normalization per sample (standard scaling)
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True)
        x = (x - mean) / (std + 1e-6)
        
        return x, y

# ============================================
# ARCHITECTURE DEFINITIONS
# ============================================
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

class HybridFireGNN(nn.Module):
    def __init__(self, in_channels=12, hidden_dim=64, dropout=0.3):
        super(HybridFireGNN, self).__init__()
        self.dropout = dropout
        
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

        # Graph Attention Bottleneck 
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
        self.gat3 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout)

        # U-Net Decoder with Skip Connections
        self.up1 = nn.ConvTranspose2d(hidden_dim, 64, 2, 2)  # 8->16
        self.conv_up1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Dropout2d(dropout)
        )
        self.up2 = nn.ConvTranspose2d(64, 32, 2, 2)  # 16->32
        self.conv_up2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Dropout2d(dropout)
        )
        self.up3 = nn.ConvTranspose2d(32, 16, 2, 2)  # 32->64
        self.final = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Grid -> Graph
        B, C_lat, H_lat, W_lat = e3.shape
        x_flat = e3.permute(0, 2, 3, 1).reshape(B * H_lat * W_lat, C_lat)
        
        # Build batched edge index
        single_edges = get_grid_topology(H_lat, W_lat, x.device)
        edge_indices = [single_edges + i * (H_lat * W_lat) for i in range(B)]
        batched_edges = torch.cat(edge_indices, dim=1)
        
        # 3 GAT layers with residual connection
        g = F.elu(self.gat1(x_flat, batched_edges))
        g = F.elu(self.gat2(g, batched_edges))
        g = F.elu(self.gat3(g, batched_edges)) + x_flat  # Residual
        
        # Graph -> Grid
        x_gnn = g.reshape(B, H_lat, W_lat, C_lat).permute(0, 3, 1, 2)
        
        # Decoder 
        d1 = self.conv_up1(torch.cat([self.up1(x_gnn), e2], dim=1))
        d2 = self.conv_up2(torch.cat([self.up2(d1), e1], dim=1))
        out = self.final(self.up3(d2))
        
        return out

# ============================================
# EVALUATION & VISUALIZATION
# ============================================
def find_optimal_threshold(model, val_loader, device):
    model.eval()
    all_probs = []
    all_targets = []
    
    print("\\nFinding optimal threshold on VALIDATION set...")
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            y_bin = (y > 0.0).float()
            
            all_probs.extend(probs.cpu().numpy().flatten())
            all_targets.extend(y_bin.cpu().numpy().flatten())
    
    y_true = np.array(all_targets)
    y_prob = np.array(all_probs)
    
    thresholds = np.arange(0.2, 0.7, 0.05)
    best_f1 = 0
    best_thresh = 0.5
    
    for thresh in thresholds:
        y_pred_t = (y_prob > thresh).astype(float)
        f1_t = f1_score(y_true, y_pred_t, zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = thresh
    
    print(f"Optimal threshold on validation set: {best_thresh:.2f} (F1={best_f1:.3f})")
    return best_thresh

def evaluate_on_test_set(model, test_loader, device, threshold):
    model.eval()
    all_probs = []
    all_targets = []
    baseline_preds = []
    
    print(f"\\nEvaluating on TEST SET with threshold={threshold:.2f}...")
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            y_bin = (y > 0.0).float()
            
            all_probs.extend(probs.cpu().numpy().flatten())
            all_targets.extend(y_bin.cpu().numpy().flatten())
            
            # Persistence baseline (PrevFireMask)
            # Normalization was applied, so > 0.0 is not correct anymore. 
            # Actually, PrevFireMask is 0 or 1. Let's un-normalize or keep it simple.
            # In WildfireDataset, x is normalized. We need to extract the original x if we want the actual baseline.
            # As a shortcut, we just check if the normalized PrevFireMask is > mean. 
            # But earlier it was standardized (x - mean)/std. 
            # A value originally 0 will be < 0, a value originally > 0 will be > 0.
            # So (x[:, 11:12, :, :] > 0) still roughly isolates fire pixels.
            prev_fire = x[:, 11:12, :, :]
            baseline_preds.extend((prev_fire > 0.0).float().cpu().numpy().flatten())
    
    y_true = np.array(all_targets)
    y_prob = np.array(all_probs)
    y_base = np.array(baseline_preds)
    
    y_pred = (y_prob > threshold).astype(float)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    iou = jaccard_score(y_true, y_pred, zero_division=0)
    auprc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    b_iou = jaccard_score(y_true, y_base, zero_division=0)
    b_prec, b_rec, b_f1, _ = precision_recall_fscore_support(y_true, y_base, average='binary', zero_division=0)
    
    y_pred_05 = (y_prob > 0.5).astype(float)
    _, _, f1_05, _ = precision_recall_fscore_support(y_true, y_pred_05, average='binary', zero_division=0)
    iou_05 = jaccard_score(y_true, y_pred_05, zero_division=0)
    
    print("\\n" + "="*70)
    print(" FINAL TEST SET RESULTS")
    print("="*70)
    print(f"{'Model':<28} | {'IoU':>6} | {'Recall':>7} | {'Prec':>6} | {'F1':>6} | {'AUPRC':>6}")
    print("-" * 70)
    # print(f"{'Persistence Baseline':<28} | {b_iou:>6.3f} | {b_rec:>7.3f} | {b_prec:>6.3f} | {b_f1:>6.3f} | {'---':>6}")
    print(f"{'Ours (thresh=0.5)':<28} | {iou_05:>6.3f} | {'---':>7} | {'---':>6} | {f1_05:>6.3f} | {'---':>6}")
    print(f"Ours (thresh={threshold:.2f})    | {iou:>6.3f} | {recall:>7.3f} | {precision:>6.3f} | {f1:>6.3f} | {auprc:>6.3f}")
    print("="*70)
    print(f"\\nROC-AUC: {roc_auc:.3f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    axes[0].plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
    axes[0].plot([0,1], [0,1], 'k--', lw=1)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
    axes[1].plot(rec_curve, prec_curve, 'g-', lw=2, label=f'PR (AUPRC = {auprc:.2f})')
    axes[1].axhline(y=precision, color='r', linestyle='--', alpha=0.5, label=f'Thresh={threshold:.2f}')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('roc_pr_curves.png', dpi=150, bbox_inches='tight')

def visualize_prediction(model, loader, device, threshold):
    model.eval()
    
    # scan for a batch with some actual fires to visualize
    found_fire = False
    for bx, by in loader:
        if by.sum() > 0:
            x, y = bx, by
            found_fire = True
            break
            
    if not found_fire:
        print("No fire pixels found in test subset. Using first batch.")
        x, y = next(iter(loader))
        
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        logits = model(x)
        pred = torch.sigmoid(logits)
    
    input_fire = x[0, 11, :, :].cpu().numpy()  # PrevFireMask
    target_fire = y[0, 0, :, :].cpu().numpy()
    predicted_prob = pred[0, 0, :, :].cpu().numpy()
    predicted_mask = (predicted_prob > threshold).astype(float)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(input_fire, cmap='Reds')
    axes[0].set_title("Input: Prev-Day Fire")
    axes[0].axis('off')
    
    axes[1].imshow(target_fire, cmap='Reds', vmin=0, vmax=1)
    axes[1].set_title("Target: Next-Day Fire")
    axes[1].axis('off')
    
    im = axes[2].imshow(predicted_prob, cmap='jet', vmin=0, vmax=1)
    axes[2].set_title("U-Net Predicted Probability")
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(predicted_mask, cmap='Reds', vmin=0, vmax=1)
    axes[3].set_title(f"U-Net Mask (Thresh {threshold:.2f})")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_sample.png', dpi=150, bbox_inches='tight')

# ============================================
# MAIN SCRIPT
# ============================================
def main():
    prepare_data()

    print("Initializing DataLoaders...")
    train_dataset = WildfireDataset(OUTPUT_DIR, split='train')
    val_dataset = WildfireDataset(OUTPUT_DIR, split='val')
    test_dataset = WildfireDataset(OUTPUT_DIR, split='test')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    model = HybridFireGNN(in_channels=12, hidden_dim=64, dropout=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    pos_weight = torch.tensor([8.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    NUM_EPOCHS = 100 # Increased to 100 epochs for deeper training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    print("="*60)
    print(" TRAINING: Hybrid ResGNN-UNet over FULL Dataset")
    print("="*60)

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_train_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            y = (y > 0.0).float()
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item()
            
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y = (y > 0.0).float()
                logits = model(x)
                loss = criterion(logits, y)
                epoch_val_loss += loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step()
        
        marker = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            marker = "  Best"
        
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}{marker}")

    # Load best model for evaluation
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    
    # Plot learning curves
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, 'b-', label='Train Loss')
    plt.plot(val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=150)

    # Evaluate
    optimal_threshold = find_optimal_threshold(model, val_loader, device)
    evaluate_on_test_set(model, test_loader, device, optimal_threshold)
    visualize_prediction(model, test_loader, device, optimal_threshold)

if __name__ == '__main__':
    main()

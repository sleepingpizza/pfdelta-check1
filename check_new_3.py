import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import csv
import sys
import argparse
import logging

# --- CONSTANTS ---
IDX_PD, IDX_PG = 0, 1
IDX_QD, IDX_QG = 2, 3
IDX_VM, IDX_VA = 4, 5
IDX_TYPE_PQ, IDX_TYPE_PV, IDX_TYPE_SLACK = 6, 7, 8

NORM_COLS = [IDX_PD, IDX_PG, IDX_QD, IDX_QG]

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--outdir", type=str, default="mae_results_multiview")
parser.add_argument("--epochs", type=int, default=1500)
parser.add_argument("--data_path", type=str, default="pfdelta_unified.npz")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_views", type=int, default=4, help="How many times to repeat data with different permutations")
parser.add_argument("--embed_dim", type=int, default=128)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--mask_ratio", type=float, default=0.5)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# --- Logger ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(args.outdir, "log.txt")), logging.StreamHandler(sys.stdout)]
)

class CSVLogger:
    def __init__(self, outdir):
        self.path = os.path.join(outdir, "metrics.csv")
        self.headers = ["epoch", "train_loss", "lr", "val_loss_augmented", "val_loss_canonical","val_vm", "val_va","val_pg","val_qg", "worst_bus_idx", "worst_bus_mse"]
        with open(self.path, 'w', newline='') as f: csv.writer(f).writerow(self.headers)
    def log(self, row_dict):
        row = [row_dict.get(h, "") for h in self.headers]
        with open(self.path, 'a', newline='') as f: csv.writer(f).writerow(row)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class PowerFlowDataset(Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data.astype(np.float32))
        self.num_buses = data.shape[1]
    def __len__(self): return self.data.shape[0]
    def __getitem__(self, idx):
        return self.data[idx]

# --- MULTI-VIEW PHYSICS COLLATE ---
def make_multiview_collate_fn(num_views=4, mask_ratio=0.5):
    def collate_fn(batch):
        # 1. Stack raw samples [B, N, 9]
        base_tokens = torch.stack(batch, dim=0) 
        B, N, C = base_tokens.shape

        tokens = base_tokens.repeat_interleave(num_views, dim=0)
        B_total = tokens.shape[0]
        
       
        perms = torch.stack([torch.randperm(N) for _ in range(B_total)], dim=0)
        
        # Apply the shuffle
        perms_expanded = perms.unsqueeze(-1).expand(-1, -1, C)
        tokens_shuffled = torch.gather(tokens, 1, perms_expanded)
        
        # Generate mask on the shuffled data
        mask_candidates = torch.zeros_like(tokens_shuffled, dtype=torch.bool)
        
        is_pq = tokens_shuffled[:, :, IDX_TYPE_PQ] > 0.5
        is_pv = tokens_shuffled[:, :, IDX_TYPE_PV] > 0.5
        is_slack = tokens_shuffled[:, :, IDX_TYPE_SLACK] > 0.5
        
        mask_candidates[:, :, IDX_VM] = is_pq
        mask_candidates[:, :, IDX_VA] = is_pq | is_pv 
        mask_candidates[:, :, IDX_QG] = is_pv | is_slack 
        mask_candidates[:, :, IDX_PG] = is_slack
        
        noise = torch.rand_like(tokens_shuffled)

        mask = (noise < mask_ratio) & mask_candidates
        return {
            "tokens": tokens_shuffled, 
            "node_ids": perms, 
            "mask": mask,
            "targets": tokens_shuffled.clone() 
        }
    return collate_fn
def make_canonical_collate_fn():
    def collate_fn(batch):
        tokens = torch.stack(batch, dim=0)
        B, N, C = tokens.shape
        
        node_ids = torch.arange(N).unsqueeze(0).expand(B, -1)
        mask_candidates = torch.zeros_like(tokens, dtype=torch.bool)

        is_pq = tokens[:, :, IDX_TYPE_PQ] > 0.5
        is_pv = tokens[:, :, IDX_TYPE_PV] > 0.5
        is_slack = tokens[:, :, IDX_TYPE_SLACK] > 0.5
        
        mask_candidates[:, :, IDX_VM] = is_pq
        mask_candidates[:, :, IDX_VA] = is_pq | is_pv   
        mask_candidates[:, :, IDX_QG] = is_pv | is_slack
        mask_candidates[:, :, IDX_PG] = is_slack
        mask = mask_candidates
        return {
            "tokens": tokens, 
            "node_ids": node_ids,
            "mask": mask,
            "targets": tokens.clone()
        }
    return collate_fn
class MAEModel(nn.Module):
    def __init__(self, token_dim, embed_dim, num_buses, num_layers=4):
        super().__init__()
        self.token_embedding = nn.Linear(token_dim, embed_dim)
        self.bus_id_embedding = nn.Embedding(num_buses, embed_dim)
        
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, token_dim)
        )

    def forward(self, x, node_ids):
        x_emb = self.token_embedding(x)
        topo_emb = self.bus_id_embedding(node_ids)
        x_in = x_emb + topo_emb
        x_enc = self.encoder(x_in) 
        out = self.decoder(x_enc)
        return out

def train_epoch(model, loader, optimizer, device, scaler = None, use_amp = False):
    model.train()
    total_loss = 0
    #scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else torch.cuda.amp.GradScaler()
    
    if use_amp:
        autocast = torch.cuda.amp.autocast
    
    else:
        class _noop:
            def __enter__(self): return None
            def __exit__(self, exc_type, exc_value, traceback): return False
        autocast = _noop
    for batch in loader:

        tokens = batch["tokens"].to(device)
        node_ids = batch["node_ids"].to(device)
        mask = batch["mask"].to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            masked_input = tokens.masked_fill(mask, 0.0)
            preds = model(masked_input, node_ids)
            
            squared_error = (preds - tokens) ** 2
            loss_tensor = squared_error * mask.float()
            loss = loss_tensor.sum() / (mask.float().sum() + 1e-8)
            
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0
    losses = {
        'Pg': 0.0, 'Qg': 0.0, 'Vm': 0.0, 'Va': 0.0, 'overall': 0.0
    }
    counts = {k : 0 for k in losses.keys()}
    sum_sq_error = None
    count_samples = 0
    
    for batch in loader:
        tokens = batch["tokens"].to(device)
        node_ids = batch["node_ids"].to(device)
        mask = batch["mask"].to(device)
        
        masked_input = tokens.masked_fill(mask, 0.0)
        preds = model(masked_input, node_ids)
        
        sq_err = (preds - tokens) ** 2

        for idx, name in [(IDX_PG, 'Pg'), (IDX_QG, 'Qg'), (IDX_VM, 'Vm'), (IDX_VA, 'Va')]:
            var_loss = 0.0
            var_mask = mask[:, :, idx].float()
            if var_mask.any():
                var_loss += (sq_err[:, :, idx] * var_mask).sum()
                losses[name] += var_loss.item()
                counts[name] += var_mask.sum().item()

        loss_tensor = sq_err * mask.float()
        losses['overall'] += loss_tensor.sum().item()
        counts['overall'] += mask.float().sum().item()
        total_loss += (loss_tensor.sum() / (mask.sum() + 1e-8)).item()
        
        # Calculate Worst Bus Error
        B, N, C = tokens.shape
        vm_mask = mask[:, :, IDX_VM].float()
        vm_err = sq_err[:, :, IDX_VM] * vm_mask 
        
        if sum_sq_error is None:
            sum_sq_error = torch.zeros(N, device=device)
            count_samples = torch.zeros(N, device=device)
            
        flat_ids = node_ids.view(-1)
        flat_err = vm_err.view(-1)
        flat_mask = vm_mask.view(-1)
        
        sum_sq_error.index_add_(0, flat_ids, flat_err)
        count_samples.index_add_(0, flat_ids, flat_mask)
    
    for k in losses.keys():
        losses[k] /= counts[k] + 1e-8
    
    #avg_loss = total_loss / len(loader)
    
    safe_count = count_samples.clamp(min=1)
    mse_per_bus = sum_sq_error / safe_count
    
    valid_buses = count_samples > 0
    if valid_buses.sum() > 0:
        worst_idx = torch.argmax(mse_per_bus * valid_buses.float()).item()
        worst_mse = mse_per_bus[worst_idx].item()
    else:
        worst_idx = -1; worst_mse = 0.0
        
    return losses, worst_idx, worst_mse

def main():
    print("--- Script Starting ---")
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    if not os.path.exists(args.data_path):
        print("ERROR: Data file not found! Run transform script first."); sys.exit(1)
    
    print("Loading Data...")
    raw_data = np.load(args.data_path)['tokens']
    np.random.shuffle(raw_data) # Shuffle for distribution check
    
    N_buses = raw_data.shape[1]
    
    split_idx = int(0.9 * len(raw_data))
    train_data = raw_data[:split_idx]
    val_data = raw_data[split_idx:]
    
    print("Computing Stats...")
    means = np.zeros(9, dtype=np.float32)
    stds = np.ones(9, dtype=np.float32)
    
    train_flat = train_data.reshape(-1, 9)
    for c in NORM_COLS:
        means[c] = train_flat[:, c].mean()
        stds[c] = train_flat[:, c].std() + 1e-8
    
    def apply_norm(arr):
        return (arr - means[None, None, :]) / stds[None, None, :]
    
    train_norm = apply_norm(train_data)
    val_norm = apply_norm(val_data)
    
    train_collate_fn = make_multiview_collate_fn(num_views=args.num_views, mask_ratio=args.mask_ratio)
    val_augmented_collate_fn = make_multiview_collate_fn(num_views=args.num_views, mask_ratio=args.mask_ratio)
    val_canonical_collate_fn = make_canonical_collate_fn()

    print(f"Effective Batch Size (Compute): {args.batch_size} * {args.num_views} = {args.batch_size * args.num_views}")
    
    train_loader = DataLoader(PowerFlowDataset(train_norm), batch_size=args.batch_size, shuffle=True, collate_fn=train_collate_fn)
    val_augmented_loader = DataLoader(PowerFlowDataset(val_norm), batch_size=args.batch_size, collate_fn=val_augmented_collate_fn)
    val_canonical_loader = DataLoader(PowerFlowDataset(val_norm), batch_size=args.batch_size, collate_fn=val_canonical_collate_fn)
    
    model = MAEModel(token_dim=9, embed_dim=args.embed_dim, num_buses=N_buses, num_layers=args.num_layers).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    logger = CSVLogger(args.outdir)
    
    print("Starting Training...")
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, scaler = scaler, use_amp = use_amp)
        val_losses_aug, w_idx_aug, w_mse_aug = validate(model, val_augmented_loader, device)
        val_losses_can, w_idx_can, w_mse_can = validate(model, val_canonical_loader, device)
        
        if epoch % 10 == 0:
            logging.info(f"Ep {epoch} | LR : {scheduler.get_last_lr()[0]:.6f} | Train Loss: {train_loss:.6f}")
            logging.info(f"  Val Aug: {val_losses_aug['overall']:.4f} | Val Can: {val_losses_can['overall']:.4f}")
            logging.info(f"  Vm: {val_losses_can['Vm']:.4f} | Va: {val_losses_can['Va']:.4f} | Pg: {val_losses_can['Pg']:.4f} | Qg: {val_losses_can['Qg']:.4f}")
            logging.info(f"  Worst Bus {w_idx_can}: {w_mse_can:.4f}")
        ##logger.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "worst_bus_idx": w_idx, "worst_bus_mse": w_mse})
        logger.log({
            "epoch": epoch,
            "train_loss": train_loss,
            #"lr": current_lr,
            "val_loss_augmented": val_losses_aug['overall'],
            "val_loss_canonical": val_losses_can['overall'],
            "val_Vm": val_losses_can['Vm'],
            "val_Va": val_losses_can['Va'],
            "val_Pg": val_losses_can['Pg'],
            "val_Qg": val_losses_can['Qg'],
            "worst_bus_idx": w_idx_can,
            "worst_bus_mse": w_mse_can
        })
        if val_losses_can['overall'] < best_loss:
            best_loss = val_losses_can['overall']
            torch.save(model.state_dict(), os.path.join(args.outdir, "best_model.pt"))

if __name__ == "__main__":
    main()
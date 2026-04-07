import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import argparse
from tqdm import tqdm
import random

from train.dataset import SpectrumDataset
from models.mace_net import SpectralEquivariantGNN
from train.losses import bipartite_matching_loss, auto_differential_spectrum_loss

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, optimizer, device, lambda_spectrum=0.3, grad_clip=1.0):
    model.train()
    epoch_bipartite = 0.0
    epoch_spectrum = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        pred_dict = model(batch)
        
        # Compute losses
        loss_bipartite, _, _ = bipartite_matching_loss(pred_dict, batch)
        loss_spectrum = auto_differential_spectrum_loss(pred_dict, batch)
        
        total_loss = loss_bipartite + lambda_spectrum * loss_spectrum
        
        total_loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        
        epoch_bipartite += loss_bipartite.item()
        epoch_spectrum += loss_spectrum.item()
        
    return epoch_bipartite / len(dataloader), epoch_spectrum / len(dataloader)

def evaluate(model, dataloader, device, lambda_spectrum=0.3):
    model.eval()
    val_bipartite = 0.0
    val_spectrum = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch.to(device)
            
            pred_dict = model(batch)
            
            loss_bipartite, _, _ = bipartite_matching_loss(pred_dict, batch)
            loss_spectrum = auto_differential_spectrum_loss(pred_dict, batch)
            
            val_bipartite += loss_bipartite.item()
            val_spectrum += loss_spectrum.item()
            
    return val_bipartite / len(dataloader), val_spectrum / len(dataloader)

def main():
    parser = argparse.ArgumentParser(description="Train E(3)-Equivariant GNN for Spectra")
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Directory with .pt files')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--k_max', type=int, default=64)
    parser.add_argument('--amp_scale', type=float, default=1e-3)
    parser.add_argument('--lambda_spectrum', type=float, default=0.3)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Simple split for demonstration (in reality, you'd want proper train/val split logic)
    # Here we just use the same data for train/val as we only have ammonia right now
    dataset = SpectrumDataset(args.data_dir)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize Model
    model = SpectralEquivariantGNN(
        node_features_in=5, # Ensure this matches one-hot embeddings shape
        hidden_dim=args.hidden_dim,
        K_max=args.k_max,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        amp_scale=args.amp_scale,
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_loss = float('inf')
    
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        train_bip, train_spec = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            lambda_spectrum=args.lambda_spectrum,
            grad_clip=args.grad_clip,
        )
        val_bip, val_spec = evaluate(
            model,
            val_loader,
            device,
            lambda_spectrum=args.lambda_spectrum,
        )
        
        val_total = val_bip + args.lambda_spectrum * val_spec
        scheduler.step(val_total)
        
        print(f"Train - Bipartite: {train_bip:.4f}, Spectrum: {train_spec:.4f}")
        print(f"Val   - Bipartite: {val_bip:.4f}, Spectrum: {val_spec:.4f}")
        
        if val_total < best_loss:
            best_loss = val_total
            print(f"New best validation loss: {best_loss:.4f}. Saving model...")
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            
    print("Training complete!")

if __name__ == "__main__":
    main()

import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
import wandb
from tqdm import tqdm

from config import cfg
from dataset import BirdDataset, get_transforms
from model import BirdModel, ModelEMA
from augs import MixupCutmix

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, ema_model, loader, optimizer, scheduler, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    augmenter = MixupCutmix(
        mixup_prob=cfg.mixup_prob, 
        cutmix_prob=cfg.cutmix_prob, 
        alpha=cfg.mixup_alpha
    )

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()

        images, target_a, target_b, lam, type = augmenter(images, labels)
        
        with autocast(enabled=cfg.use_amp):
            outputs = model(images)
            if type != 'none':
                loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
            else:
                loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    
        if ema_model:
            ema_model.update(model)
  
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(target_a).sum().item()

        if scheduler is not None:
            scheduler.step()
            
        pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def valid_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Valid", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def run_fold(fold, df, train_idx, val_idx):
    print(f"\n--- Iniciando Fold: {fold+1}/{cfg.n_folds} ---")

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    
    train_ds = BirdDataset(df_train, cfg.cropped_dir, transform=get_transforms('train', cfg.img_size))
    val_ds = BirdDataset(df_val, cfg.cropped_dir, transform=get_transforms('valid', cfg.img_size))
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, 
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, 
                            num_workers=cfg.num_workers, pin_memory=True)

    model = BirdModel(cfg.model.name, cfg.model.num_classes, pretrained=cfg.model.pretrained)
    model.to(cfg.device)

    ema_model = ModelEMA(model, decay=cfg.model.ema_decay, device=cfg.device)
    ema_model.set(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=cfg.use_amp)

    total_steps = len(train_loader) * cfg.epochs
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=total_steps, 
        eta_min=cfg.min_lr
    )

    run_name = f"{cfg.model.name}_fold{fold+1}"
    wandb.init(
        project=cfg.project_name, 
        name=run_name, 
        group=cfg.exp_name,
        config=cfg.__dict__,
        reinit=True
    )
    
    best_acc = 0.0

    for epoch in range(cfg.epochs):
        train_loss, train_acc = train_one_epoch(
            model, ema_model, train_loader, optimizer, scheduler, criterion, cfg.device, scaler
        )
        
        val_loss, val_acc = valid_one_epoch(ema_model.module, val_loader, criterion, cfg.device)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val (EMA) Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0]
        })

        if val_acc > best_acc:
            best_acc = val_acc
            save_name = f"{cfg.model.name}_fold{fold+1}_best.pth"
            save_path = os.path.join(cfg.output_dir, save_name)
            torch.save(ema_model.module.state_dict(), save_path)
            print(f" Mejor modelo guardado (Acc: {best_acc:.4f})")
            
    wandb.finish()

    del model, ema_model, optimizer, scheduler, scaler, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()
    
    return best_acc

if __name__ == "__main__":
    wandb.login()
    seed_everything(cfg.seed)

    print(f"Leyendo CSV: {cfg.train_csv_path}")
    df = pd.read_csv(cfg.train_csv_path)

    num_classes = df['cls'].nunique()
    cfg.model.num_classes = num_classes
    print(f"Detectadas {num_classes} clases.")

    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['cls'])):
        acc = run_fold(fold, df, train_idx, val_idx)
        fold_scores.append(acc)

    print("\n" + "="*40)
    print(f"CV RESULTS")
    print(f"Scores: {fold_scores}")
    print(f"Average Accuracy: {np.mean(fold_scores):.4f}")
    print("="*40)
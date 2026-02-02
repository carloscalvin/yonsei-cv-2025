import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import cfg
from dataset import BirdDataset, get_transforms
from model import BirdModel

def load_models(device):
    models = []
    print(f"Cargando {cfg.n_folds} modelos para Ensemble...")
    
    for fold in range(1, cfg.n_folds + 1):
        model = BirdModel(cfg.model.name, cfg.model.num_classes, pretrained=False)
        model.to(device)
        model.eval()
        
        weight_name = f"{cfg.model.name}_fold{fold}_best.pth"
        weight_path = os.path.join(cfg.output_dir, weight_name)
        
        if os.path.exists(weight_path):
            state_dict = torch.load(weight_path, map_location=device)
            model.load_state_dict(state_dict)
            models.append(model)
            print(f" -> Modelo Fold {fold} cargado correctamente.")
        else:
            print(f" [!] ALERTA: No se encontró {weight_path}. Se saltará este fold.")
            
    return models

@torch.no_grad()
def inference():
    device = cfg.device

    print(f"Leyendo CSV de Test: {cfg.test_csv_path}")
    test_df = pd.read_csv(cfg.test_csv_path)

    test_dataset = BirdDataset(
        test_df, 
        cfg.cropped_dir, 
        transform=get_transforms('valid', cfg.img_size)
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    models = load_models(device)
    
    if len(models) == 0:
        print("Error: No se cargó ningún modelo. Revisa las rutas.")
        return

    print("Iniciando inferencia (Ensemble)...")
    final_preds = []
    final_confs = []

    for images, _ in tqdm(test_loader, desc="TTA Inference"):
        images = images.to(device)
        images_flip = torch.flip(images, dims=[3])
        batch_probs = torch.zeros(images.shape[0], cfg.model.num_classes).to(device)

        for model in models:
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            outputs_flip = model(images_flip)
            probs_flip = torch.softmax(outputs_flip, dim=1)
            avg_probs = (probs + probs_flip) / 2.0            
            batch_probs += avg_probs

        batch_probs /= len(models)
        batch_confs, predicted_labels = torch.max(batch_probs, 1)

        final_preds.extend(predicted_labels.cpu().numpy())
        final_confs.extend(batch_confs.cpu().numpy())

    test_df['cls'] = final_preds
    if cfg.debug:
        test_df['confidence'] = final_confs

    submission_path = os.path.join(cfg.output_dir, "submission.csv")
    test_df.to_csv(submission_path, index=False)
    
    print("\n" + "="*40)
    print(f"Inferencia terminada.")
    print(f"Archivo guardado en: {submission_path}")
    print("Muestra de las primeras 5 filas:")
    print(test_df.head())
    print("="*40)

if __name__ == "__main__":
    inference()
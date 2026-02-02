import os
import cv2
import argparse
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from config import cfg
from dataset import get_transforms
from model import BirdModel

def parse_args():
    parser = argparse.ArgumentParser(description="Herramienta de Diagnóstico Visual para Inferencia")
    parser.add_argument("filename", type=str, help="Nombre EXACTO del archivo (ej: 'test_0028.jpg')")
    return parser.parse_args()

def load_resources():
    print("Cargando CSVs y Modelos (Ensemble)...")
    if not os.path.exists(cfg.train_csv_path):
        raise FileNotFoundError(f"No se encuentra {cfg.train_csv_path}")
        
    train_df = pd.read_csv(cfg.train_csv_path)
    
    models = []
    device = cfg.device
    
    for fold in range(1, cfg.n_folds + 1):
        weight_path = os.path.join(cfg.output_dir, f"{cfg.model.name}_fold{fold}_best.pth")
        if os.path.exists(weight_path):
            model = BirdModel(cfg.model.name, cfg.model.num_classes, pretrained=False)
            model.load_state_dict(torch.load(weight_path, map_location=device))
            model.to(device)
            model.eval()
            models.append(model)
            
    if not models:
        raise FileNotFoundError("No se encontraron pesos de modelos en 'outputs/'.")
        
    return train_df, models

def get_image_tensor(filename):
    paths_to_try = [
        os.path.join(cfg.cropped_dir, filename),
        os.path.join(cfg.images_dir, filename)
    ]
    
    img_path = next((p for p in paths_to_try if os.path.exists(p)), None)
    
    if not img_path:
        return None, None
        
    image = cv2.imread(img_path)
    if image is None: return None, None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    original_vis = image.copy()
    
    transform = get_transforms('valid', cfg.img_size)
    augmented = transform(image=image)
    tensor = augmented['image'].unsqueeze(0)
    
    return tensor, original_vis

def get_train_reference_image(cls_id, train_df):
    rows = train_df[train_df['cls'] == cls_id]
    if rows.empty:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    
    filename = rows.iloc[0]['filename']
    path = os.path.join(cfg.cropped_dir, filename)
    if not os.path.exists(path):
        path = os.path.join(cfg.images_dir, filename)
        
    img = cv2.imread(path)
    if img is None:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def show_top5_grid(test_filename, test_img, top_probs, top_classes, train_df):
    fig = plt.figure(figsize=(18, 8))

    ax_test = plt.subplot2grid((2, 6), (0, 0), rowspan=2, colspan=3)
    ax_test.imshow(test_img)
    ax_test.set_title(f"TEST: {test_filename}", fontsize=16, fontweight='bold')
    ax_test.axis('off')

    for i in range(5):
        if i >= len(top_classes): break
        
        cls_id = top_classes[i]
        prob = top_probs[i]

        ref_img = get_train_reference_image(cls_id, train_df)

        row = 0 if i < 3 else 1
        col = 3 + (i % 3)
        
        ax = plt.subplot2grid((2, 6), (row, col))
        ax.imshow(ref_img)
        
        color = 'green' if prob > 0.5 else 'black'
        title_text = f"#{i+1}: Class {cls_id}\nConf: {prob:.2%}"
        ax.set_title(title_text, color=color, fontsize=10, fontweight='bold')
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

def main():
    args = parse_args()
    
    path_check_1 = os.path.join(cfg.cropped_dir, args.filename)
    path_check_2 = os.path.join(cfg.images_dir, args.filename)
    
    if not os.path.exists(path_check_1) and not os.path.exists(path_check_2):
        print(f"Error: El archivo '{args.filename}' no existe en las carpetas de imágenes.")
        return

    train_df, models = load_resources()
    
    print(f"Analizando: {args.filename} ...")
    
    tensor, img_vis = get_image_tensor(args.filename)
    if tensor is None:
        print("Error: No se pudo leer la imagen (formato corrupto o error de CV2).")
        return

    avg_probs = torch.zeros(1, cfg.model.num_classes).to(cfg.device)
    tensor = tensor.to(cfg.device)
    tensor_flip = torch.flip(tensor, dims=[3])

    with torch.no_grad():
        for model in models:
            p1 = torch.softmax(model(tensor), dim=1)
            p2 = torch.softmax(model(tensor_flip), dim=1)
            avg_probs += (p1 + p2) / 2.0
    
    avg_probs /= len(models)

    topk_probs, topk_indices = torch.topk(avg_probs, 5)
    topk_probs = topk_probs.cpu().numpy()[0]
    topk_indices = topk_indices.cpu().numpy()[0]

    print("\n" + "="*40)
    print(f" RESULTADOS PARA: {args.filename}")
    print("="*40)
    print(f"{'RANK':<5} | {'CLASE':<10} | {'CONFIANZA':<10}")
    print("-" * 35)
    for i, (p, c) in enumerate(zip(topk_probs, topk_indices)):
        print(f"#{i+1:<4} | {c:<10} | {p:.4f}")
    print("-" * 35)

    show_top5_grid(args.filename, img_vis, topk_probs, topk_indices, train_df)

if __name__ == "__main__":
    main()
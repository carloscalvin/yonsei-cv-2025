import os
import shutil
import pandas as pd
from tqdm import tqdm
from configs.config import cfg

def organize_by_species():
    review_dir = os.path.join(os.path.dirname(cfg.cropped_dir), "manual_review_species")
    
    if os.path.exists(review_dir):
        print(f"El directorio {review_dir} ya existe. Se agregarán/sobrescribirán archivos.")
    else:
        os.makedirs(review_dir)
        print(f"Directorio creado: {review_dir}")

    print(f"Leendo etiquetas desde: {cfg.train_csv_path}")
    df = pd.read_csv(cfg.train_csv_path)

    copied_count = 0
    missing_count = 0

    print(f"Organizando {len(df)} registros en carpetas por especie...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        filename = row['filename']
        cls_id = row['cls']

        src_path = os.path.join(cfg.cropped_dir, filename)
        
        if os.path.exists(src_path):
            class_dir = os.path.join(review_dir, str(cls_id))
            os.makedirs(class_dir, exist_ok=True)
            dst_path = os.path.join(class_dir, filename)
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        else:
            missing_count += 1

    print("\n" + "="*40)
    print("REPORTE DE ORGANIZACIÓN")
    print("="*40)
    print(f"Carpetas generadas en: {review_dir}")
    print(f"Imágenes organizadas:  {copied_count}")
    print(f"Imágenes no encontradas: {missing_count}")
    print("="*40)

if __name__ == "__main__":
    organize_by_species()
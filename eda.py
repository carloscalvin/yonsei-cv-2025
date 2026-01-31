import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
from config import cfg

def run_eda():
    print(f"Cargando datos desde: {cfg.train_csv_path}")
    df_train = pd.read_csv(cfg.train_csv_path)
    df_test = pd.read_csv(cfg.test_csv_path)
    
    print(f"Train shape: {df_train.shape}")
    print(f"Test shape: {df_test.shape}")

    num_classes = df_train['cls'].nunique()
    print(f"Número de especies detectadas: {num_classes}")

    plt.figure(figsize=(12, 6))
    sns.countplot(x=df_train['cls'], palette='viridis')
    plt.title(f"Distribución de Clases (Total: {num_classes})")
    plt.xlabel("ID Clase")
    plt.ylabel("Conteo")
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "eda_class_distribution.png"))
    plt.show()

    print("Analizando dimensiones de imágenes (esto puede tardar un poco)...")
    widths, heights = [], []
    
    for filename in df_train['filename']:
        img_path = os.path.join(cfg.images_dir, filename)
        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        else:
            print(f"Imagen no encontrada: {filename}")

    plt.figure(figsize=(10, 5))
    plt.scatter(widths, heights, alpha=0.3, c='blue')
    plt.title(f"Dimensiones de Imágenes (Promedio: {np.mean(widths):.0f}x{np.mean(heights):.0f})")
    plt.xlabel("Ancho")
    plt.ylabel("Alto")
    plt.grid(True, alpha=0.3)

    plt.axvline(cfg.img_size, color='red', linestyle='--', label=f'Input Modelo ({cfg.img_size})')
    plt.axhline(cfg.img_size, color='red', linestyle='--')
    plt.legend()
    
    plt.savefig(os.path.join(cfg.output_dir, "eda_image_sizes.png"))
    plt.show()

    print("\n" + "="*40)
    print("REPORTE DE ESTADÍSTICAS DEL DATASET")
    print("="*40)

    class_counts = df_train['cls'].value_counts()
    print("\n  Distribución de Clases:")
    print(f" - Mínimo de imágenes en una clase: {class_counts.min()}")
    print(f" - Máximo de imágenes en una clase: {class_counts.max()}")
    print(f" - Promedio de imágenes por clase:  {class_counts.mean():.2f}")
    print(f" - Desviación estándar:             {class_counts.std():.2f}")

    if widths and heights:
        w_arr = np.array(widths)
        h_arr = np.array(heights)
        ar_arr = w_arr / h_arr # Aspect Ratio
        total_imgs = len(w_arr)
        
        non_square = np.sum(w_arr != h_arr)
        exact_500 = np.sum((w_arr == 500) & (h_arr == 500))
        w_perc = np.percentile(w_arr, [25, 50, 75])
        h_perc = np.percentile(h_arr, [25, 50, 75])

        print("\nDimensiones y Geometría:")
        print(f" - Ancho (Min / Media / Max): {w_arr.min()} / {w_arr.mean():.0f} / {w_arr.max()}")
        print(f" - Alto  (Min / Media / Max): {h_arr.min()} / {h_arr.mean():.0f} / {h_arr.max()}")
        print(f" - Aspect Ratio Promedio:     {ar_arr.mean():.2f}")
        print("-" * 20)
        print(f" - Imágenes NO cuadradas:     {non_square} ({non_square/total_imgs:.1%} del total)")
        print(f" - Imágenes exactas 500x500:  {exact_500} ({exact_500/total_imgs:.1%} del total)")
        print("-" * 20)
        print(f" - Percentiles Ancho [25, 50, 75]: {w_perc}")
        print(f" - Percentiles Alto  [25, 50, 75]: {h_perc}")

        small_imgs = np.sum((w_arr < cfg.img_size) | (h_arr < cfg.img_size))
        if small_imgs > 0:
            print(f" ALERTA: Hay {small_imgs} imágenes con algún lado menor al input ({cfg.img_size}px).")
            print("    (Estas imágenes sufrirán 'upscaling' y podrían perderse detalles o pixelarse).")
    
    print("="*40 + "\n")

    print("EDA Finalizado. Gráficos guardados en 'outputs/'")

if __name__ == "__main__":
    run_eda()
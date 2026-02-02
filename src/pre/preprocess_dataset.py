import os
import cv2
import glob
from ultralytics import YOLO
from tqdm import tqdm
from configs.config import cfg

def pad_to_square(img, color=(0, 0, 0)):
    h, w = img.shape[:2]
    if h == w:
        return img
    max_side = max(h, w)
    delta_w = max_side - w
    delta_h = max_side - h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    new_img = cv2.copyMakeBorder(
        img, top, bottom, left, right, 
        cv2.BORDER_CONSTANT, value=color
    )
    return new_img

def smart_process_image(model, image_path, output_folder, target_size=224):
    filename = os.path.basename(image_path)
    img = cv2.imread(image_path)
    
    if img is None:
        return

    results = model(img, verbose=False)
    boxes = results[0].boxes
    bird_box = None
    max_conf = 0

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == 14: #14=bird 
            if conf > max_conf:
                max_conf = conf
                bird_box = box.xyxy[0].cpu().numpy()

    crop = img
    
    if bird_box is not None:
        x1, y1, x2, y2 = map(int, bird_box)
        h, w, _ = img.shape
        margin = 0.25
        
        box_h = y2 - y1
        box_w = x2 - x1

        x1 = max(0, int(x1 - box_w * margin))
        y1 = max(0, int(y1 - box_h * margin))
        x2 = min(w, int(x2 + box_w * margin))
        y2 = min(h, int(y2 + box_h * margin))
        
        crop = img[y1:y2, x1:x2]
    else:
        print(f"No bird in {filename}, using full image.")
        pass

    if crop.size == 0: return

    square_img = pad_to_square(crop)
    final_img = cv2.resize(square_img, (target_size, target_size), interpolation=cv2.INTER_AREA)

    save_path = os.path.join(output_folder, filename)
    cv2.imwrite(save_path, final_img)

def preprocess():
    print(f"Cargando modelo YOLO...")
    model_path = 'outputs/yolov8n.pt' 
    if not os.path.exists(model_path):
        model = YOLO('yolov8n.pt') 
    else:
        model = YOLO(model_path)

    extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(cfg.images_dir, ext)))
    
    print(f"Procesando {len(image_files)} imágenes en: {cfg.images_dir}")
    print(f"Guardando resultados en: {cfg.cropped_dir}")

    for img_path in tqdm(image_files):
        smart_process_image(model, img_path, cfg.cropped_dir, target_size=cfg.img_size)
    
    print("¡Proceso terminado!")
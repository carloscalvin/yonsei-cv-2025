from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO('outputs/yolov8n.pt')

def smart_crop_test(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = model(img)
    
    boxes = results[0].boxes
    bird_box = None
    
    max_conf = 0
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == 14: # 14 bird
            if conf > max_conf:
                max_conf = conf
                bird_box = box.xyxy[0].cpu().numpy() # [x1, y1, x2, y2]
    
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
        
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        plt.figure(figsize=(8,8))
        plt.imshow(img_rgb)
        plt.title(f"Pájaro detectado - Conf: {max_conf:.2f}")
        plt.show()
        
        crop = img[y1:y2, x1:x2]
        return crop
    else:
        print("No se detectó pájaro en esta imagen.")
        return img

smart_crop_test("datasets/images/train_0002.jpg")
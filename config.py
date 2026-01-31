from types import SimpleNamespace
import torch
import os

cfg = SimpleNamespace(**{})

cfg.project_name = "yonsei-cv-2025"
cfg.seed = 42
cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.num_workers = 0

cfg.root_dir = os.getcwd()
cfg.images_dir = os.path.join(cfg.root_dir, "datasets", "images")
cfg.train_csv_path = os.path.join(cfg.root_dir, "datasets", "train_anno.csv") 
cfg.test_csv_path = os.path.join(cfg.root_dir, "datasets", "test_subm.csv")
cfg.output_dir = os.path.join(cfg.root_dir, "outputs")

os.makedirs(cfg.output_dir, exist_ok=True)

cfg.batch_size = 32
cfg.img_size = 224
cfg.val_fold_size = 0.2

model_cfg = SimpleNamespace(**{})
model_cfg.name = "tf_efficientnet_b0_ns"
model_cfg.pretrained = True
model_cfg.num_classes = 50

cfg.model = model_cfg
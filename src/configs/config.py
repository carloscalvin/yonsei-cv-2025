from types import SimpleNamespace
import torch
import os

cfg = SimpleNamespace(**{})

cfg.project_name = "yonsei-cv-2025"
cfg.exp_name = "resnet34_torchvision_initial_dataset-run9"
cfg.seed = 42
cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.num_workers = 0
cfg.debug = False

cfg.root_dir = os.getcwd()
cfg.images_dir = os.path.join(cfg.root_dir, "datasets", "images")
cfg.cropped_dir = os.path.join(cfg.root_dir, "datasets", "images_cropped_square")
cfg.train_csv_path = os.path.join(cfg.root_dir, "datasets", "train_anno.csv") 
cfg.test_csv_path = os.path.join(cfg.root_dir, "datasets", "test_subm.csv")
cfg.output_dir = os.path.join(cfg.root_dir, "outputs")

os.makedirs(cfg.output_dir, exist_ok=True)
os.makedirs(cfg.cropped_dir, exist_ok=True)

cfg.batch_size = 128
cfg.img_size = 448
cfg.n_folds = 5

cfg.epochs = 50
cfg.lr = 1e-4
cfg.min_lr = 1e-6
cfg.weight_decay = 1e-4
cfg.max_grad_norm = 1.0
cfg.use_amp = True

cfg.mixup_prob = 0.0
cfg.cutmix_prob = 0.0
cfg.mixup_alpha = 1.0

model_cfg = SimpleNamespace(**{})
model_cfg.name = "resnet34_torchvision"
model_cfg.pretrained = True
model_cfg.num_classes = 50
model_cfg.ema_decay= 0.997

cfg.model = model_cfg
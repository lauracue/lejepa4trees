# -*- coding: utf-8 -*-
"""
Created on Fri May 29 12:28:26 2026

@author: cuel001
"""

# -*- coding: utf-8 -*-
"""
Fully supervised training using the cropped test dataset.

Created on Thu May 28 12:47:17 2026

@author: cuel001
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from sklearn.metrics import f1_score
import timm
import wandb
import hydra
import tqdm

from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from torchvision.ops import MLP

from dataloader import JPGDataset


class ViTEncoder(nn.Module):

    def __init__(self, num_classes=20, crop_size=128):

        super().__init__()

        self.backbone = timm.create_model(
            "resnet18",
            pretrained=True,
            num_classes=512,
            drop_path_rate=0.1,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):

        feat = self.backbone(x)
        out = self.head(feat)

        return out


@hydra.main(version_base=None)
def main(cfg: DictConfig):

    TEST_FOLDER = "./data/correcting_geospatial_data/cropped_images"

    jpg_files = list(Path(TEST_FOLDER).glob("*.jpg"))

    print(f"Found {len(jpg_files)} cropped images")

    wandb.init(project="LeJEPA_supervised_pretrained_True", config=dict(cfg))

    torch.manual_seed(0)

    # ---------------------------------------------------
    # DATASET
    # ---------------------------------------------------

    dataset = JPGDataset(
        jpg_files=jpg_files,
        crop_size=cfg.crop_size
    )

    # split into train/test
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size

    train_ds, test_ds = torch.utils.data.random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(0)
    )

    train = DataLoader(
        train_ds,
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    test = DataLoader(
        test_ds,
        batch_size=cfg.bs,
        shuffle=False,
        num_workers=0,
    )

    # ---------------------------------------------------
    # MODEL
    # ---------------------------------------------------

    net = ViTEncoder(
        num_classes=cfg.num_classes,
        crop_size=cfg.crop_size
    ).to("cuda")

    # ---------------------------------------------------
    # OPTIMIZER
    # ---------------------------------------------------

    opt = torch.optim.AdamW(
        net.parameters(),
        lr=cfg.lr,
        weight_decay=5e-2,
    )

    warmup_steps = len(train)
    total_steps = len(train) * cfg.epochs

    s1 = LinearLR(
        opt,
        start_factor=0.01,
        total_iters=warmup_steps
    )

    s2 = CosineAnnealingLR(
        opt,
        T_max=total_steps - warmup_steps,
        eta_min=1e-5
    )

    scheduler = SequentialLR(
        opt,
        schedulers=[s1, s2],
        milestones=[warmup_steps]
    )

    scaler = GradScaler(enabled=True)

    # ---------------------------------------------------
    # TRAINING
    # ---------------------------------------------------

    for epoch in range(cfg.epochs):

        net.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for x, y in tqdm.tqdm(train, total=len(train)):

            x = x.to("cuda", non_blocking=True)
            y = y.to("cuda", non_blocking=True)

            with autocast("cuda", dtype=torch.bfloat16):

                logits = net(x[:,0])

                loss = F.cross_entropy(logits, y)

            opt.zero_grad()

            scaler.scale(loss).backward()

            scaler.step(opt)

            scaler.update()

            scheduler.step()

            # metrics
            preds = logits.argmax(1)

            running_correct += (preds == y).sum().item()
            running_total += y.size(0)

            running_loss += loss.item()

            wandb.log({
                "train/loss": loss.item(),
            })

        train_acc = running_correct / running_total
        train_loss = running_loss / len(train)

        # ---------------------------------------------------
        # EVALUATION
        # ---------------------------------------------------
        
        net.eval()
        
        correct = 0
        total = 0
        test_loss = 0.0
        
        all_preds = []
        all_targets = []
        
        with torch.inference_mode():
        
            for x, y in test:
        
                x = x.to("cuda", non_blocking=True)
                y = y.to("cuda", non_blocking=True)
        
                with autocast("cuda", dtype=torch.bfloat16):
        
                    logits = net(x[:,0])
        
                    loss = F.cross_entropy(logits, y)
        
                preds = logits.argmax(1)
        
                correct += (preds == y).sum().item()
        
                total += y.size(0)
        
                test_loss += loss.item()
        
                # store predictions for F1
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        test_acc = correct / total
        test_loss = test_loss / len(test)
        
        # ---------------------------------------------------
        # F1 SCORES
        # ---------------------------------------------------
        
        per_class_f1 = f1_score(
            all_targets,
            all_preds,
            average=None,
            labels=list(range(cfg.num_classes)),
            zero_division=0,
        )
        
        macro_f1 = f1_score(
            all_targets,
            all_preds,
            average="macro",
            zero_division=0,
        )
        
        weighted_f1 = f1_score(
            all_targets,
            all_preds,
            average="weighted",
            zero_division=0,
        )
        
        # ---------------------------------------------------
        # PRINT METRICS
        # ---------------------------------------------------
        
        print(
            f"Epoch {epoch} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} | "
            f"test_acc={test_acc:.4f} | "
            f"macro_f1={macro_f1:.4f}"
        )
        
        print("Per-class F1 scores:")
        
        for cls_idx, f1 in enumerate(per_class_f1):
            print(f"Class {cls_idx}: F1 = {f1:.4f}")
        
        # ---------------------------------------------------
        # WANDB LOGGING
        # ---------------------------------------------------
        
        log_dict = {
            "epoch": epoch,
            "train/acc": train_acc,
            "train/loss_epoch": train_loss,
            "test/acc": test_acc,
            "test/loss": test_loss,
            "test/f1_macro": macro_f1,
            "test/f1_weighted": weighted_f1,
        }
        
        # add per-class f1
        for cls_idx, f1 in enumerate(per_class_f1):
            log_dict[f"test/f1_class_{cls_idx}"] = f1
        
        wandb.log(log_dict)


if __name__ == "__main__":

    cfg = OmegaConf.create({

        "bs": 256,
        "lr": 2e-4,
        "epochs": 50,
        "crop_size": 128,

        # IMPORTANT:
        # set this to the number of classes
        "num_classes": 20,
    })

    main(cfg)
